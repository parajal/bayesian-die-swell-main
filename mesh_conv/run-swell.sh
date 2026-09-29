#!/bin/bash
#SBATCH --job-name=oldroyd_uavg_sweep
#SBATCH --output=%j.out
#SBATCH --partition=mech-cem.cpu.q
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=100G

module purge
module load Umbrella/2024 foss/2025a gmsh/4.15.0-foss-2025a imkl/2025.1.0  # imkl: PARDISO

export LD_LIBRARY_PATH="${MKLROOT:-}/lib/intel64:${EBROOTFLEXIBLAS:-}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_ENABLE_INSTRUCTIONS=AVX2   # fast MKL kernels on AMD nodes

ROOTDIR="$PWD"
lambda=1
beta=0.11111111111111111
eta0=1.0
uavg_values=(0.125 0.25 0.5)
deltat_values=(0.2 0.1 0.05)   # U_avg * deltat = 0.08

OUTROOT="$ROOTDIR/sweep_oldroyd_uavg"
SUMMARY="$OUTROOT/swell_summary.txt"
mkdir -p "$OUTROOT"
echo "# U_avg deltat Wi last_step max_height end_height" > "$SUMMARY"

for k in "${!uavg_values[@]}"; do
  uavg=${uavg_values[$k]}
  deltat=${deltat_values[$k]}
  wi=$(awk -v u="$uavg" -v l="$lambda" 'BEGIN{print 4*u*l}')
  run_dir="$OUTROOT/oldroyd_lam_${lambda//./p}_beta_${beta//./p}_uavg_${uavg//./p}"
  echo "=== U_avg = $uavg  deltat = $deltat  Wi = $wi  ($run_dir)"

  if [ ! -f "$run_dir/curve4_y.txt" ]; then
    mkdir -p "$run_dir"
    cp "$ROOTDIR/mesh_2D_bc.igo" "$ROOTDIR/mesh_inlet_2D_bc.igo" "$run_dir/"
    cat > "$run_dir/input.txt" << EOF
&comppar
model        = 2
lambda       = $lambda
mobility     = 0.0
betav        = $beta
eta_0        = $eta0
U_avg        = $uavg
planar       = .false.
deltat       = $deltat
numtimesteps = 1000
dx_box       = 0.2
dx_wall      = 0.025
dx_inlet     = 0.2
vtkevery     = 100
stop_when_steady      = .true.
steady_height_tol     = 1.0e-05
steady_end_max_tol    = 5.0e-03
steady_height_steps   = 10
steady_min_steps      = 80
steady_round_decimals = 0
/
EOF
    (cd "$run_dir" && "$ROOTDIR/extrudate_swell" < input.txt > simulation.out 2>&1)
  fi

  if [ ! -f "$run_dir/curve4_y.txt" ]; then
    echo "$uavg $deltat $wi FAILED" >> "$SUMMARY"
    continue
  fi
  # last "step time max_height end_height max/initial" line of the solver output
  last=$(awk 'NF==5 && $1 ~ /^[0-9]+$/ {l=$1" "$3" "$4} END{print l}' "$run_dir/simulation.out")
  echo "$uavg $deltat $wi $last" >> "$SUMMARY"
done

cat "$SUMMARY"
