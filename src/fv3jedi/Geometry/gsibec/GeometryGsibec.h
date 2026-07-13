/*
 * (C) Copyright 2026 UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "eckit/mpi/Comm.h"

#include "atlas/field.h"
#include "atlas/functionspace.h"
#include "atlas/grid.h"

#include "ijedi/Geometry/base/GeometryBase.h"

namespace eckit
{
  class Configuration;
}

namespace ijedi
{

  // GSIBEC (background error covariance) geometry that builds an atlas FunctionSpace
  // directly from a YAML grid description. Mirrors saber::interpolation::Geometry:
  // a GSI-matching custom grid + checkerboard partitioner path.
  class GeometryGsibec : public GeometryBase
  {
   public:
    GeometryGsibec(const eckit::Configuration &, const eckit::mpi::Comm &,
                   eckit::LocalConfiguration &, atlas::FunctionSpace &,
                   atlas::FieldSet &, bool &, int &);
    std::vector<double> verticalCoord(std::string &) const override;

   private:
    void print(std::ostream &) const override;
    const eckit::mpi::Comm & comm_;
    int numLevels_ = 1;
    std::size_t halo_ = 1;
    atlas::Grid grid_;
    atlas::grid::Partitioner partitioner_;
    atlas::FunctionSpace functionSpace_;
  };

}  // namespace ijedi
