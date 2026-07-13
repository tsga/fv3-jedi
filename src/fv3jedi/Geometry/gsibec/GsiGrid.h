/*
 * (C) Copyright 2026 UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <string>
#include <vector>

#include "atlas/field.h"
#include "atlas/functionspace.h"
#include "atlas/grid.h"

#include "eckit/config/Configuration.h"
#include "eckit/mpi/Comm.h"

// Local copy of saber::interpolation's GSI-matching grid helper, vendored into ijedi so the atlas
// geometry does not depend on a saber change while that lands upstream. Keep in sync with
// saber/src/saber/interpolation/Geometry.cc.

namespace ijedi
{
  // Config keys selecting the GSI-matching grid and partitioner.
  extern const std::string GsiGridKey;
  extern const std::string GsiPartitionerKey;

  // Compute a south-to-north "checkerboard" MPI partition of a regular grid, matching GSI's layout
  // (GSI orders points/partitions south-to-north, atlas north-to-south). nbands must divide ntasks.
  std::vector<int> computeS2NCheckerboardPartition(const atlas::RegularGrid & rg,
                                                   const int ntasks, const int nbands);

  // Build an atlas grid + function space replicating the GSI grid and MPI partition (but not GSI's
  // per-task grid-point ordering). Driven by the GsiGridKey / GsiPartitionerKey sub-configs.
  void setupGsiMatchingGrid(const eckit::Configuration & config,
                            const eckit::mpi::Comm & comm,
                            atlas::Grid & grid,
                            atlas::FunctionSpace & functionSpace,
                            atlas::FieldSet & fieldSet);

}  // namespace ijedi
