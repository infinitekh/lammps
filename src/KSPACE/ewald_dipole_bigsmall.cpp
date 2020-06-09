/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Julien Tranchida (SNL), Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "ewald_dipole_bigsmall.h"
#include <mpi.h>
#include <cstring>
#include <cmath>
#include <cassert>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "domain.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "group.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

#define SMALL 0.00001

/* ---------------------------------------------------------------------- */

EwaldDipoleBigsmall::EwaldDipoleBigsmall(LAMMPS *lmp) : Ewald(lmp),
  tk(NULL), vc(NULL)
{
  ewaldflag = dipoleflag = 1;
  group_group_enable = 0;
  tk = NULL;
  vc = NULL;
}

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

EwaldDipoleBigsmall::~EwaldDipoleBigsmall()
{
  memory->destroy(tk);
  memory->destroy(vc);
}

/* ----------------------------------------------------------------------
   called once before run
------------------------------------------------------------------------- */

void EwaldDipoleBigsmall::init()
{
  if (comm->me == 0) {
    if (screen) fprintf(screen,"EwaldDipoleBigsmall initialization ...\n");
    if (logfile) fprintf(logfile,"EwaldDipoleBigsmall initialization ...\n");
  }

  // error check

  dipoleflag = atom->mu?1:0;
  qsum_qsq(0); // q[i] might not be declared ?

  if (dipoleflag && q2)
    error->all(FLERR,"Cannot (yet) use charges with Kspace style EwaldDipoleBigsmall");

  // no triclinic ewald dipole (yet)

  triclinic_check();

  triclinic = domain->triclinic;
  if (triclinic)
    error->all(FLERR,"Cannot (yet) use EwaldDipoleBigsmall with triclinic box");

  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use EwaldDipoleBigsmall with 2d simulation");

  if (!atom->mu) error->all(FLERR,"Kspace style requires atom attribute mu");

  if (dipoleflag && strcmp(update->unit_style,"electron") == 0)
    error->all(FLERR,"Cannot (yet) use 'electron' units with dipoles");

  if (slabflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use nonperiodic boundaries with EwaldDipoleBigsmall");
  if (slabflag) {
    if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
        domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR,"Incorrect boundaries with slab EwaldDipoleBigsmall");
  }

  // compute two charge force

  two_charge();

  // extract short-range Coulombic cutoff from pair style

  triclinic = domain->triclinic;
  if (triclinic)
    error->all(FLERR,"Cannot yet use triclinic cells with EwaldDipoleBigsmall");

  pair_check();

  int itmp;
  double *p_cutoff = (double *) force->pair->extract("cut_coul",itmp);
  if (p_cutoff == NULL)
    error->all(FLERR,"KSpace style is incompatible with Pair style");
  double cutoff = *p_cutoff;

  // kspace TIP4P not yet supported
  // qdist = offset only for TIP4P fictitious charge

  //qdist = 0.0;
  if (tip4pflag)
    error->all(FLERR,"Cannot yet use TIP4P with EwaldDipoleBigsmall");

	biggroup = group->find("big");
	if (biggroup == -1) //  
		error->all(FLERR,"No big group");
	biggroupbit = group->bitmask[biggroup];
	ibiggroupbit = group->inversemask[biggroup];
	bign = group->count(biggroup);
  // compute musum & musqsum and warn if no dipole

  scale = 1.0;
  qqrd2e = force->qqrd2e;
  musum_musq();
  natoms_original = atom->natoms;

  // set accuracy (force units) from accuracy_relative or accuracy_absolute

  if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
  else accuracy = accuracy_relative * two_charge_force;

  // setup K-space resolution

  bigint natoms = atom->natoms;

  // use xprd,yprd,zprd even if triclinic so grid size is the same
  // adjust z dimension for 2d slab EwaldDipoleBigsmall
  // 3d EwaldDipoleBigsmall just uses zprd since slab_volfactor = 1.0

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;

  // make initial g_ewald estimate
  // based on desired accuracy and real space cutoff
  // fluid-occupied volume used to estimate real-space error
  // zprd used rather than zprd_slab

  if (!gewaldflag) {
    if (accuracy <= 0.0)
      error->all(FLERR,"KSpace accuracy must be > 0");

    // initial guess with old method

    g_ewald = accuracy*sqrt(natoms*cutoff*xprd*yprd*zprd) / (2.0*mu2);
    if (g_ewald >= 1.0) g_ewald = (1.35 - 0.15*log(accuracy))/cutoff;
    else g_ewald = sqrt(-log(g_ewald)) / cutoff;

    // try Newton solver

    double g_ewald_new =
      NewtonSolve(g_ewald,cutoff,natoms,xprd*yprd*zprd,mu2);
    if (g_ewald_new > 0.0) g_ewald = g_ewald_new;
    else error->warning(FLERR,"Ewald/disp Newton solver failed, "
                        "using old method to estimate g_ewald");
  }

  // setup EwaldDipoleBigsmall coefficients so can print stats

  setup();

  // final RMS accuracy

  double lprx = rms(kxmax_orig,xprd,natoms,q2);
  double lpry = rms(kymax_orig,yprd,natoms,q2);
  double lprz = rms(kzmax_orig,zprd_slab,natoms,q2);
  double lpr = sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0);
  double q2_over_sqrt = q2 / sqrt(natoms*cutoff*xprd*yprd*zprd_slab);
  double spr = 2.0 *q2_over_sqrt * exp(-g_ewald*g_ewald*cutoff*cutoff);
  double tpr = estimate_table_accuracy(q2_over_sqrt,spr);
  double estimated_accuracy = sqrt(lpr*lpr + spr*spr + tpr*tpr);

  // stats

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen,"  G vector (1/distance) = %g\n",g_ewald);
      fprintf(screen,"  estimated absolute RMS force accuracy = %g\n",
              estimated_accuracy);
      fprintf(screen,"  estimated relative force accuracy = %g\n",
              estimated_accuracy/two_charge_force);
      fprintf(screen,"  KSpace vectors: actual max1d max3d = %d %d %d\n",
              kcount,kmax,kmax3d);
      fprintf(screen,"                  kxmax kymax kzmax  = %d %d %d\n",
              kxmax,kymax,kzmax);
    }
    if (logfile) {
      fprintf(logfile,"  G vector (1/distance) = %g\n",g_ewald);
      fprintf(logfile,"  estimated absolute RMS force accuracy = %g\n",
              estimated_accuracy);
      fprintf(logfile,"  estimated relative force accuracy = %g\n",
              estimated_accuracy/two_charge_force);
      fprintf(logfile,"  KSpace vectors: actual max1d max3d = %d %d %d\n",
              kcount,kmax,kmax3d);
      fprintf(logfile,"                  kxmax kymax kzmax  = %d %d %d\n",
              kxmax,kymax,kzmax);
    }
  }
}

/* ----------------------------------------------------------------------
   adjust EwaldDipoleBigsmall coeffs, called initially and whenever volume has changed
------------------------------------------------------------------------- */

void EwaldDipoleBigsmall::setup()
{
  // volume-dependent factors

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  // adjustment of z dimension for 2d slab EwaldDipoleBigsmall
  // 3d EwaldDipoleBigsmall just uses zprd since slab_volfactor = 1.0

  double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  unitk[0] = 2.0*MY_PI/xprd;
  unitk[1] = 2.0*MY_PI/yprd;
  unitk[2] = 2.0*MY_PI/zprd_slab;

  int kmax_old = kmax;

  if (kewaldflag == 0) {

    // determine kmax
    // function of current box size, accuracy, G_ewald (short-range cutoff)

    bigint natoms = atom->natoms;
    double err;
    kxmax = 1;
    kymax = 1;
    kzmax = 1;

    // set kmax in 3 directions to respect accuracy

    err = rms_dipole(kxmax,xprd,natoms);
    while (err > accuracy) {
      kxmax++;
      err = rms_dipole(kxmax,xprd,natoms);
    }

    err = rms_dipole(kymax,yprd,natoms);
    while (err > accuracy) {
      kymax++;
      err = rms_dipole(kymax,yprd,natoms);
    }

    err = rms_dipole(kzmax,zprd,natoms);
    while (err > accuracy) {
      kzmax++;
      err = rms_dipole(kzmax,zprd,natoms);
    }

    kmax = MAX(kxmax,kymax);
    kmax = MAX(kmax,kzmax);
    kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;

    double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
    double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
    double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
    gsqmx = MAX(gsqxmx,gsqymx);
    gsqmx = MAX(gsqmx,gsqzmx);

    kxmax_orig = kxmax;
    kymax_orig = kymax;
    kzmax_orig = kzmax;

  } else {

    kxmax = kx_ewald;
    kymax = ky_ewald;
    kzmax = kz_ewald;

    kxmax_orig = kxmax;
    kymax_orig = kymax;
    kzmax_orig = kzmax;

    kmax = MAX(kxmax,kymax);
    kmax = MAX(kmax,kzmax);
    kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;

    double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
    double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
    double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
    gsqmx = MAX(gsqxmx,gsqymx);
    gsqmx = MAX(gsqmx,gsqzmx);
  }

  gsqmx *= 1.00001;

  // if size has grown, reallocate k-dependent and nlocal-dependent arrays

  if (kmax > kmax_old) {
    deallocate();
    allocate();
    group_allocate_flag = 0;

    memory->destroy(ek);
    memory->destroy(tk);
    memory->destroy(oldx);
    memory->destroy(oldmu);
    memory->destroy(oldF);
    memory->destroy(oldT);
    memory->destroy(delF);
    memory->destroy(delT);
    memory->destroy(vc);
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    nmax = atom->nmax;
    memory->create(ek,bign,3,"ewald_dipole_bigsmall:ek");
    memory->create(tk,bign,3,"ewald_dipole_bigsmall:tk");
    memory->create(oldx,bign,6,"ewald_dipole_bigsmall:oldx");
    memory->create(oldmu,bign,6,"ewald_dipole_bigsmall:oldmu");
    memory->create(oldF,bign,12,"ewald_dipole_bigsmall:oldF");
    memory->create(oldT,bign,12,"ewald_dipole_bigsmall:oldT");
    memory->create(delF,bign,12,"ewald_dipole_bigsmall:delF");
    memory->create(delT,bign,12,"ewald_dipole_bigsmall:delT");
    memory->create(vc,kmax3d,6,"ewald_dipole_bigsmall:tk");
    memory->create3d_offset(cs,-kmax,kmax,3,bign,"ewald_dipole_bigsmall:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,bign,"ewald_dipole_bigsmall:sn");
    kmax_created = kmax;
  }

  // pre-compute EwaldDipoleBigsmall coefficients

  coeffs();
}

/* ----------------------------------------------------------------------
   compute dipole RMS accuracy for a dimension
------------------------------------------------------------------------- */

double EwaldDipoleBigsmall::rms_dipole(int km, double prd, bigint natoms)
{
  if (natoms == 0) natoms = 1;   // avoid division by zero

  // error from eq.(46), Wang et al., JCP 115, 6351 (2001)

  double value = 8*MY_PI*mu2*g_ewald/volume *
    sqrt(2*MY_PI*km*km*km/(15.0*natoms)) *
    exp(-MY_PI*MY_PI*km*km/(g_ewald*g_ewald*prd*prd));

  return value;
}

/* ----------------------------------------------------------------------
   compute the EwaldDipoleBigsmall long-range force, energy, virial
------------------------------------------------------------------------- */

void EwaldDipoleBigsmall::compute(int eflag, int vflag)
{
  int i,j,k;
  const double g3 = g_ewald*g_ewald*g_ewald;
	
/* 	int nfix = modify->nfix;
 * 	Fix* fix = modify->fix;
 * 	FixSRD* fix_srd;
 * 	int biggroupbit;
 * 	int bign;
 * 
 * 
 * 	if (fix_srd == NULL) {
 * 		for (i = 0; i < nfix; i++) {
 * 			if(	strncmp(fix[i]->style,"srd",strlen("srd")) ==0 ) {
 * 				fix_srd = fix[i]; 
 * 				biggroupbit = fix_srd->biggroupbit;
 * 				bign = fix_srd->bign;
 * 			}
 * 		}
 * 	}
 */

  // set energy/virial flags

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = evflag_atom = eflag_global = vflag_global =
         eflag_atom = vflag_atom = 0;

  // if atom count has changed, update qsum and qsqsum

  if (atom->natoms != natoms_original) {
    musum_musq();
    natoms_original = atom->natoms;
  }

  // return if there are no charges

  if (musqsum == 0.0) return;

  // extend size of per-atom arrays if necessary

  if (group->count(biggroup) > bign) {
    memory->destroy(ek);
    memory->destroy(tk);
    memory->destroy(oldx);
    memory->destroy(oldmu);
    memory->destroy(delF);
    memory->destroy(delT);
    memory->destroy(vc);
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    bign = group->count(biggroup);
    memory->create(ek,bign,3,"ewald_dipole_bigsmall:ek");
    memory->create(tk,bign,3,"ewald_dipole_bigsmall:tk");
    memory->create(oldx,bign,6,"ewald_dipole_bigsmall:oldx");
    memory->create(oldmu,bign,6,"ewald_dipole_bigsmall:oldmu");
    memory->create(oldF,bign,3,"ewald_dipole_bigsmall:oldF");
    memory->create(oldT,bign,3,"ewald_dipole_bigsmall:oldT");
    memory->create(delF,bign,12,"ewald_dipole_bigsmall:delF");
    memory->create(delT,bign,12,"ewald_dipole_bigsmall:delT");
    memory->create(vc,kmax3d,6,"ewald_dipole_bigsmall:tk");
    memory->create3d_offset(cs,-kmax,kmax,3,bign,"ewald_dipole_bigsmall:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,bign,"ewald_dipole_bigsmall:sn");
    kmax_created = kmax;
  }

  // partial structure factors on each processor
  // total structure factor by summing over procs

  eik_dot_r();

  MPI_Allreduce(sfacrl,sfacrl_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_all,kcount,MPI_DOUBLE,MPI_SUM,world);

  // K-space portion of electric field
  // double loop over K-vectors and local atoms
  // perform per-atom calculations if needed

  double **f = atom->f;
  double **t = atom->torque;
  double **mu = atom->mu;
  int nlocal = atom->nlocal;

  int kx,ky,kz;
  double cypz,sypz,exprl,expim;
  double partial,partial_peratom;
  double vcik[6];
  double mudotk;

	int bigi = 0;
	for (i = 0; i < nlocal; i++) {
		// only big particle.
		if (atom->mask[i] & ibiggroupbit)  continue;
    ek[bigi][0] = ek[bigi][1] = ek[bigi][2] = 0.0;
    tk[bigi][0] = tk[bigi][1] = tk[bigi][2] = 0.0;
		bigi = bigi +1;
#ifdef NDEBUG
		if ( bigi == bign ) break;
#endif
	} 
	assert( bigi == bign );

  for (k = 0; k < kcount; k++) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];
    for (j = 0; j<6; j++) vc[k][j] = 0.0;

		int bigi = 0;
		for (i = 0; i < nlocal; i++) {
			// only big particle.
			if (atom->mask[i] & ibiggroupbit)  continue;

      for (j = 0; j<6; j++) vcik[j] = 0.0;

      // re-evaluating mu dot k
      mudotk = mu[i][0]*kx*unitk[0] + mu[i][1]*ky*unitk[1] + mu[i][2]*kz*unitk[2];

      // calculating  re and im of exp(i*k*ri)

      cypz = cs[ky][1][bigi]*cs[kz][2][bigi] - sn[ky][1][bigi]*sn[kz][2][bigi];
      sypz = sn[ky][1][bigi]*cs[kz][2][bigi] + cs[ky][1][bigi]*sn[kz][2][bigi];
      exprl = cs[kx][0][bigi]*cypz - sn[kx][0][bigi]*sypz;
      expim = sn[kx][0][bigi]*cypz + cs[kx][0][bigi]*sypz;

      // taking im of struct_fact x exp(i*k*ri) (for force calc.)

      partial = (mudotk)*(expim*sfacrl_all[k] - exprl*sfacim_all[k]);
      ek[bigi][0] += partial * eg[k][0];
      ek[bigi][1] += partial * eg[k][1];
      ek[bigi][2] += partial * eg[k][2];

      // compute field for torque calculation

      partial_peratom = exprl*sfacrl_all[k] + expim*sfacim_all[k];
      tk[bigi][0] += partial_peratom * eg[k][0];
      tk[bigi][1] += partial_peratom * eg[k][1];
      tk[bigi][2] += partial_peratom * eg[k][2];

      // total and per-atom virial correction (dipole only)

      vc[k][0] += vcik[0] = -(partial_peratom * mu[i][0] * eg[k][0]);
      vc[k][1] += vcik[1] = -(partial_peratom * mu[i][1] * eg[k][1]);
      vc[k][2] += vcik[2] = -(partial_peratom * mu[i][2] * eg[k][2]);
      vc[k][3] += vcik[3] = -(partial_peratom * mu[i][0] * eg[k][1]);
      vc[k][4] += vcik[4] = -(partial_peratom * mu[i][0] * eg[k][2]);
      vc[k][5] += vcik[5] = -(partial_peratom * mu[i][1] * eg[k][2]);

      // taking re-part of struct_fact x exp(i*k*ri)
      // (for per-atom energy and virial calc.)

      if (evflag_atom) {
        if (eflag_atom) eatom[i] += mudotk*ug[k]*partial_peratom;
        if (vflag_atom)
          for (j = 0; j < 6; j++)
            vatom[i][j] += (ug[k]*mudotk*vg[k][j]*partial_peratom - vcik[j]);
      }
			bigi = bigi +1;
#ifdef NDEBUG
			if ( bigi == bign ) break;
#endif
		} 
		assert( bigi == bign );
  }

  // force and torque calculation

  const double muscale = qqrd2e * scale;

	bigi = 0;
	for (i = 0; i < nlocal; i++) {
		// only big particle.
		if (atom->mask[i] & ibiggroupbit)  continue;
    f[i][0] += muscale * ek[bigi][0];
    f[i][1] += muscale * ek[bigi][1];
    if (slabflag != 2) f[i][2] += muscale * ek[i][2];
    t[i][0] -= muscale * (mu[i][1]*tk[bigi][2] - mu[i][2]*tk[bigi][1]);
    t[i][1] -= muscale * (mu[i][2]*tk[bigi][0] - mu[i][0]*tk[bigi][2]);
    if (slabflag != 2) t[i][2] -= muscale * (mu[i][0]*tk[bigi][1] - mu[i][1]*tk[bigi][0]);
		bigi = bigi +1;
#ifdef NDEBUG
		if ( bigi == bign ) break;
#endif
	} 
	assert( bigi == bign );

  // sum global energy across Kspace vevs and add in volume-dependent term
  // taking the re-part of struct_fact_i x struct_fact_j
  // subtracting self energy and scaling

  if (eflag_global) {
    for (k = 0; k < kcount; k++) {
      energy += ug[k] * (sfacrl_all[k]*sfacrl_all[k] +
                         sfacim_all[k]*sfacim_all[k]);
    }
    energy -= musqsum*2.0*g3/3.0/MY_PIS;
    energy *= muscale;
  }

  // global virial

  if (vflag_global) {
    double uk;
    for (k = 0; k < kcount; k++) {
      uk = ug[k] * (sfacrl_all[k]*sfacrl_all[k] + sfacim_all[k]*sfacim_all[k]);
      for (j = 0; j < 6; j++) virial[j] += uk*vg[k][j] - vc[k][j];
    }
    for (j = 0; j < 6; j++) virial[j] *= muscale;
  }

  // per-atom energy/virial
  // energy includes self-energy correction

  if (evflag_atom) {
    if (eflag_atom) {
			int bigi = 0;
			for (i = 0; i < nlocal; i++) {
				// only big particle.
				if (atom->mask[i] & ibiggroupbit)  continue;
        eatom[i] -= (mu[i][0]*mu[i][0] + mu[i][1]*mu[i][1] + mu[i][2]*mu[i][2])
          *2.0*g3/3.0/MY_PIS;
        eatom[i] *= muscale;
				bigi = bigi +1;
#ifdef NDEBUG
				if ( bigi == bign ) break;
#endif
			} 
			assert( bigi == bign );
    }

		if (vflag_atom) {
			int bigi = 0;
			for (i = 0; i < nlocal; i++) {
				// only big particle.
				if (atom->mask[i] & ibiggroupbit)  continue;
				for (j = 0; j < 6; j++) vatom[i][j] *= muscale;
				bigi = bigi +1;
#ifdef NDEBUG
				if ( bigi == bign ) break;
#endif
			} 
			assert( bigi == bign );
		}
  }

  // 2d slab correction

  if (slabflag == 1) slabcorr();
}

/* ----------------------------------------------------------------------
   compute the struc. factors and mu dot k products
------------------------------------------------------------------------- */

void EwaldDipoleBigsmall::eik_dot_r()
{
  int i,k,l,m,n,ic;
  double cstr1,sstr1,cstr2,sstr2,cstr3,sstr3,cstr4,sstr4;
  double sqk,clpm,slpm;
  double mux, muy, muz;
  double mudotk;

  double **x = atom->x;
  double **mu = atom->mu;
  int nlocal = atom->nlocal;
	biggroup = group->find("big");
	biggroupbit = group->bitmask[biggroup];
	ibiggroupbit = group->inversemask[biggroup];
	bign = group->count(biggroup);

  n = 0;
  mux = muy = muz = 0.0;

  // loop on different k-directions
  // loop on n kpoints and nlocal atoms

  // (k,0,0), (0,l,0), (0,0,m)

  // loop 1: k=1, l=1, m=1
  // define first val. of cos and sin

  for (ic = 0; ic < 3; ic++) {
    sqk = unitk[ic]*unitk[ic];
    if (sqk <= gsqmx) {
      cstr1 = 0.0;
      sstr1 = 0.0;
			int bigi = 0;
      for (i = 0; i < nlocal; i++) {
				// only big particle.
				if (atom->mask[i] & ibiggroupbit)  continue;
        cs[0][ic][bigi] = 1.0;
        sn[0][ic][bigi] = 0.0;
        cs[1][ic][bigi] = cos(unitk[ic]*x[i][ic]);
        sn[1][ic][bigi] = sin(unitk[ic]*x[i][ic]);
        cs[-1][ic][bigi] = cs[1][ic][bigi];
        sn[-1][ic][bigi] = -sn[1][ic][bigi];
        mudotk = (mu[bigi][ic]*unitk[ic]);
        cstr1 += mudotk*cs[1][ic][bigi];
        sstr1 += mudotk*sn[1][ic][bigi];
				bigi = bigi + 1;
#ifdef NDEBUG
				if ( bigi == bign ) break;
#endif
      } 
			assert( bigi == bign );
      sfacrl[n] = cstr1;
      sfacim[n++] = sstr1;
    }
  }

  // loop 2: k>1, l>1, m>1

  for (m = 2; m <= kmax; m++) {
    for (ic = 0; ic < 3; ic++) {
      sqk = m*unitk[ic] * m*unitk[ic];
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
				int bigi = 0;
				for (i = 0; i < nlocal; i++) {
					// only big particle.
					if (atom->mask[i] & ibiggroupbit)  continue;
					cs[m][ic][bigi] = cs[m-1][ic][bigi]*cs[1][ic][bigi] -
            sn[m-1][ic][bigi]*sn[1][ic][bigi];
          sn[m][ic][bigi] = sn[m-1][ic][bigi]*cs[1][ic][bigi] +
            cs[m-1][ic][bigi]*sn[1][ic][bigi];
          cs[-m][ic][bigi] = cs[m][ic][bigi];
          sn[-m][ic][bigi] = -sn[m][ic][bigi];
          mudotk = (mu[bigi][ic]*m*unitk[ic]);
          cstr1 += mudotk*cs[m][ic][bigi];
          sstr1 += mudotk*sn[m][ic][bigi];
					bigi = bigi +1;
#ifdef NDEBUG
					if ( bigi == bign ) break;
#endif
				} 
				assert( bigi == bign );
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
      }
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
				int bigi = 0;
				for (i = 0; i < nlocal; i++) {
					// only big particle.
					if (atom->mask[i] & ibiggroupbit)  continue;
          mux = mu[i][0];
          muy = mu[i][1];

          // dir 1: (k,l,0)
          mudotk = (mux*k*unitk[0] + muy*l*unitk[1]);
          cstr1 += mudotk*(cs[k][0][bigi]*cs[l][1][bigi]-sn[k][0][bigi]*sn[l][1][bigi]);
          sstr1 += mudotk*(sn[k][0][bigi]*cs[l][1][bigi]+cs[k][0][bigi]*sn[l][1][bigi]);

          // dir 2: (k,-l,0)
          mudotk = (mux*k*unitk[0] - muy*l*unitk[1]);
          cstr2 += mudotk*(cs[k][0][bigi]*cs[l][1][bigi]+sn[k][0][bigi]*sn[l][1][bigi]);
          sstr2 += mudotk*(sn[k][0][bigi]*cs[l][1][bigi]-cs[k][0][bigi]*sn[l][1][bigi]);
					bigi = bigi +1;
#ifdef NDEBUG
					if ( bigi == bign ) break;
#endif
				} 
				assert( bigi == bign );
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
				int bigi = 0;
				for (i = 0; i < nlocal; i++) {
					// only big particle.
					if (atom->mask[i] & ibiggroupbit)  continue;
          muy = mu[i][1];
          muz = mu[i][2];

          // dir 1: (0,l,m)
          mudotk = (muy*l*unitk[1] + muz*m*unitk[2]);
          cstr1 += mudotk*(cs[l][1][bigi]*cs[m][2][bigi] - sn[l][1][bigi]*sn[m][2][bigi]);
          sstr1 += mudotk*(sn[l][1][bigi]*cs[m][2][bigi] + cs[l][1][bigi]*sn[m][2][bigi]);

          // dir 2: (0,l,-m)
          mudotk = (muy*l*unitk[1] - muz*m*unitk[2]);
          cstr2 += mudotk*(cs[l][1][bigi]*cs[m][2][bigi]+sn[l][1][bigi]*sn[m][2][bigi]);
          sstr2 += mudotk*(sn[l][1][bigi]*cs[m][2][bigi]-cs[l][1][bigi]*sn[m][2][bigi]);
					bigi = bigi +1;
#ifdef NDEBUG
					if ( bigi == bign ) break;
#endif
				} 
				assert( bigi == bign );
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
				int bigi = 0;
				for (i = 0; i < nlocal; i++) {
					// only big particle.
					if (atom->mask[i] & ibiggroupbit)  continue;
          mux = mu[i][0];
          muz = mu[i][2];

          // dir 1: (k,0,m)
          mudotk = (mux*k*unitk[0] + muz*m*unitk[2]);
          cstr1 += mudotk*(cs[k][0][bigi]*cs[m][2][bigi]-sn[k][0][bigi]*sn[m][2][bigi]);
          sstr1 += mudotk*(sn[k][0][bigi]*cs[m][2][bigi]+cs[k][0][bigi]*sn[m][2][bigi]);

          // dir 2: (k,0,-m)
          mudotk = (mux*k*unitk[0] - muz*m*unitk[2]);
          cstr2 += mudotk*(cs[k][0][bigi]*cs[m][2][bigi]+sn[k][0][bigi]*sn[m][2][bigi]);
          sstr2 += mudotk*(sn[k][0][bigi]*cs[m][2][bigi]-cs[k][0][bigi]*sn[m][2][bigi]);
					bigi = bigi +1;
#ifdef NDEBUG
					if ( bigi == bign ) break;
#endif
				} 
				assert( bigi == bign );
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
          (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          cstr3 = 0.0;
          sstr3 = 0.0;
          cstr4 = 0.0;
          sstr4 = 0.0;
					int bigi = 0;
					for (i = 0; i < nlocal; i++) {
						// only big particle.
						if (atom->mask[i] & ibiggroupbit)  continue;
            mux = mu[i][0];
            muy = mu[i][1];
            muz = mu[i][2];

            // dir 1: (k,l,m)
            mudotk = (mux*k*unitk[0] + muy*l*unitk[1] + muz*m*unitk[2]);
            clpm = cs[l][1][bigi]*cs[m][2][bigi] - sn[l][1][bigi]*sn[m][2][bigi];
            slpm = sn[l][1][bigi]*cs[m][2][bigi] + cs[l][1][bigi]*sn[m][2][bigi];
            cstr1 += mudotk*(cs[k][0][bigi]*clpm - sn[k][0][bigi]*slpm);
            sstr1 += mudotk*(sn[k][0][bigi]*clpm + cs[k][0][bigi]*slpm);

            // dir 2: (k,-l,m)
            mudotk = (mux*k*unitk[0] - muy*l*unitk[1] + muz*m*unitk[2]);
            clpm = cs[l][1][bigi]*cs[m][2][bigi] + sn[l][1][bigi]*sn[m][2][bigi];
            slpm = -sn[l][1][bigi]*cs[m][2][bigi] + cs[l][1][bigi]*sn[m][2][bigi];
            cstr2 += mudotk*(cs[k][0][bigi]*clpm - sn[k][0][bigi]*slpm);
            sstr2 += mudotk*(sn[k][0][bigi]*clpm + cs[k][0][bigi]*slpm);

            // dir 3: (k,l,-m)
            mudotk = (mux*k*unitk[0] + muy*l*unitk[1] - muz*m*unitk[2]);
            clpm = cs[l][1][bigi]*cs[m][2][bigi] + sn[l][1][bigi]*sn[m][2][bigi];
            slpm = sn[l][1][bigi]*cs[m][2][bigi] - cs[l][1][bigi]*sn[m][2][bigi];
            cstr3 += mudotk*(cs[k][0][bigi]*clpm - sn[k][0][bigi]*slpm);
            sstr3 += mudotk*(sn[k][0][bigi]*clpm + cs[k][0][bigi]*slpm);

            // dir 4: (k,-l,-m)
            mudotk = (mux*k*unitk[0] - muy*l*unitk[1] - muz*m*unitk[2]);
            clpm = cs[l][1][bigi]*cs[m][2][bigi] - sn[l][1][bigi]*sn[m][2][bigi];
            slpm = -sn[l][1][bigi]*cs[m][2][bigi] - cs[l][1][bigi]*sn[m][2][bigi];
            cstr4 += mudotk*(cs[k][0][bigi]*clpm - sn[k][0][bigi]*slpm);
            sstr4 += mudotk*(sn[k][0][bigi]*clpm + cs[k][0][bigi]*slpm);
						bigi = bigi +1;
#ifdef NDEBUG
						if ( bigi == bign ) break;
#endif
					} 
					assert( bigi == bign );
          sfacrl[n] = cstr1;
          sfacim[n++] = sstr1;
          sfacrl[n] = cstr2;
          sfacim[n++] = sstr2;
          sfacrl[n] = cstr3;
          sfacim[n++] = sstr3;
          sfacrl[n] = cstr4;
          sfacim[n++] = sstr4;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D EwaldDipoleBigsmall if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
------------------------------------------------------------------------- */

void EwaldDipoleBigsmall::slabcorr()
{
  // compute local contribution to global dipole moment

  double dipole = 0.0;
  double **mu = atom->mu;
  int nlocal = atom->nlocal;
	int bigi = 0;
	for (int i = 0; i < nlocal; i++) {
		// only big particle.
		if (atom->mask[i] & ibiggroupbit)  continue;
		dipole += mu[i][2];
		bigi = bigi +1;
#ifdef NDEBUG
		if ( bigi == bign ) break;
#endif
	} 
	assert( bigi == bign );
  // sum local contributions to get global dipole moment

  double dipole_all;
  MPI_Allreduce(&dipole,&dipole_all,1,MPI_DOUBLE,MPI_SUM,world);

  // need to make non-neutral systems and/or
  //  per-atom energy translationally invariant

  if (eflag_atom || fabs(qsum) > SMALL) {

    error->all(FLERR,"Cannot (yet) use kspace slab correction with "
      "long-range dipoles and non-neutral systems or per-atom energy");
  }

  // compute corrections

  const double e_slabcorr = MY_2PI*(dipole_all*dipole_all/12.0)/volume;
  const double qscale = qqrd2e * scale;

  if (eflag_global) energy += qscale * e_slabcorr;

  // per-atom energy

  if (eflag_atom) {
    double efact = qscale * MY_2PI/volume/12.0;
    for (int i = 0; i < nlocal; i++)
      eatom[i] += efact * mu[i][2]*dipole_all;
  }

  // add on torque corrections

  if (atom->torque) {
    double ffact = qscale * (-4.0*MY_PI/volume);
    double **mu = atom->mu;
    double **torque = atom->torque;
		int bigi = 0;
		for (int i = 0; i < nlocal; i++) {
			// only big particle.
			if (atom->mask[i] & ibiggroupbit)  continue;
      torque[i][0] += ffact * dipole_all * mu[i][1];
      torque[i][1] += -ffact * dipole_all * mu[i][0];
			bigi = bigi +1;
#ifdef NDEBUG
			if ( bigi == bign ) break;
#endif
		} 
		assert( bigi == bign );
  }
}

/* 
 * Big Particle is heavy than small particle. 
 *
 * So that Big particles move only slightly on the small particle's time scale.
 */
void EwaldDipoleBigsmall::smallApprox()
{
  double **x = atom->x;
  double **f = atom->f;
  double **t = atom->torque;
  double **mu = atom->mu;
  int nlocal = atom->nlocal;
	
	const double muscale = qqrd2e * scale;
	double dx[3], dmu[3];
	energy = old_energy;

  for (int i = 0; i < nlocal; i++) {
		dx[0]  = x[i][0]  - oldx[i][0];
		dx[1]  = x[i][1]  - oldx[i][1];
		dx[2]  = x[i][2]  - oldx[i][2];
		dmu[0] = mu[i][0] - oldmu[i][0];
		dmu[1] = mu[i][1] - oldmu[i][1];
		dmu[2] = mu[i][2] - oldmu[i][2];
		
		energy -= muscale * ek[i][0] * dx[0];
		energy -= muscale * ek[i][1] * dx[1];
		energy -= muscale * ek[i][2] * dx[2];
		energy += muscale * tk[i][0] * dmu[0];
		energy += muscale * tk[i][1] * dmu[1];
		energy += muscale * tk[i][2] * dmu[2];
		
		f[i][0] += oldF[i][0];
		f[i][1] += oldF[i][1];
		f[i][2] += oldF[i][2];
		f[i][0] += muscale * delF[i][0] * dx[0];
		f[i][1] += muscale * delF[i][1] * dx[1];
		f[i][2] += muscale * delF[i][2] * dx[2];
		f[i][0] += muscale * delF[i][3] * dmu[0];
		f[i][1] += muscale * delF[i][4] * dmu[1];
		f[i][2] += muscale * delF[i][5] * dmu[2];

		t[i][0] += oldT[i][0];
		t[i][1] += oldT[i][1];
		t[i][2] += oldT[i][2];
		t[i][0] += muscale * delT[i][0] * dx[0];
		t[i][1] += muscale * delT[i][1] * dx[1];
		t[i][2] += muscale * delT[i][2] * dx[2];
		t[i][0] += muscale * delT[i][3] * dmu[0];
		t[i][1] += muscale * delT[i][4] * dmu[1];
		t[i][2] += muscale * delT[i][5] * dmu[2];
	}
	
}
/* ----------------------------------------------------------------------
   compute musum,musqsum,mu2
   called initially, when particle count changes, when dipoles are changed
------------------------------------------------------------------------- */

void EwaldDipoleBigsmall::musum_musq()
{
  const int nlocal = atom->nlocal;
	biggroup = group->find("big");
	biggroupbit = group->bitmask[biggroup];
	ibiggroupbit = group->inversemask[biggroup];
	bigint bign = group->count(biggroup);

  musum = musqsum = mu2 = 0.0;
  if (atom->mu_flag) {
    double** mu = atom->mu;
    double musum_local(0.0), musqsum_local(0.0);

		int bigi = 0;
		for (int i = 0; i < nlocal; i++) {
			// only big particle.
			if (atom->mask[i] & ibiggroupbit)  continue;
      musum_local += mu[i][0] + mu[i][1] + mu[i][2];
      musqsum_local += mu[i][0]*mu[i][0] + mu[i][1]*mu[i][1] + mu[i][2]*mu[i][2];
			bigi = bigi +1;
#ifdef NDEBUG
			if ( bigi == bign ) break;
#endif
		} 
		assert( bigi == bign );

    MPI_Allreduce(&musum_local,&musum,1,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&musqsum_local,&musqsum,1,MPI_DOUBLE,MPI_SUM,world);

    mu2 = musqsum * force->qqrd2e;
  }

  if (mu2 == 0 && comm->me == 0)
    error->all(FLERR,"Using kspace solver EwaldDipoleBigsmall on system with no dipoles");
}

/* ----------------------------------------------------------------------
   Newton solver used to find g_ewald for LJ systems
------------------------------------------------------------------------- */

double EwaldDipoleBigsmall::NewtonSolve(double x, double Rc,
                              bigint natoms, double vol, double b2)
{
  double dx,tol;
  int maxit;

  maxit = 10000; //Maximum number of iterations
  tol = 0.00001; //Convergence tolerance

  //Begin algorithm

  for (int i = 0; i < maxit; i++) {
    dx = f(x,Rc,natoms,vol,b2) / derivf(x,Rc,natoms,vol,b2);
    x = x - dx; //Update x
    if (fabs(dx) < tol) return x;
    if (x < 0 || x != x) // solver failed
      return -1;
  }
  return -1;
}

/* ----------------------------------------------------------------------
 Calculate f(x)
 ------------------------------------------------------------------------- */

double EwaldDipoleBigsmall::f(double x, double Rc, bigint natoms, double vol, double b2)
{
  double a = Rc*x;
  double f = 0.0;

  double rg2 = a*a;
  double rg4 = rg2*rg2;
  double rg6 = rg4*rg2;
  double Cc = 4.0*rg4 + 6.0*rg2 + 3.0;
  double Dc = 8.0*rg6 + 20.0*rg4 + 30.0*rg2 + 15.0;
  f = (b2/(sqrt(vol*powint(x,4)*powint(Rc,9)*natoms)) *
    sqrt(13.0/6.0*Cc*Cc + 2.0/15.0*Dc*Dc - 13.0/15.0*Cc*Dc) *
    exp(-rg2)) - accuracy;

  return f;
}

/* ----------------------------------------------------------------------
 Calculate numerical derivative f'(x)
 ------------------------------------------------------------------------- */

double EwaldDipoleBigsmall::derivf(double x, double Rc,
                         bigint natoms, double vol, double b2)
{
  double h = 0.000001;  //Derivative step-size
  return (f(x + h,Rc,natoms,vol,b2) - f(x,Rc,natoms,vol,b2)) / h;
}

