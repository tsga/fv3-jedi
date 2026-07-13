/*
 * (C) Copyright 2026 UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "ijedi/Geometry/gsibec/GeometryGsibec.h"

#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "atlas/array.h"
#include "atlas/field.h"
#include "atlas/functionspace.h"
#include "atlas/grid.h"
#include "atlas/mesh.h"

#include "eckit/config/Configuration.h"
#include "eckit/exception/Exceptions.h"

#include "oops/util/FunctionSpaceHelpers.h"
#include "oops/util/Logger.h"

#include "ijedi/Geometry/gsibec/GsiGrid.h"

namespace ijedi {

// -----------------------------------------------------------------------------------------------

GeometryGsibec::GeometryGsibec(const eckit::Configuration &conf,
                               const eckit::mpi::Comm &comm,
                               eckit::LocalConfiguration & /*geomVariables*/,
                               atlas::FunctionSpace &functionSpace,
                               atlas::FieldSet &geomFields,
                               bool &levelsAreTopDown, int &numLevels)
  : comm_(comm) {
  oops::Log::trace() << "GeometryGsibec constructor starting" << std::endl;

  levelsAreTopDown = conf.getBool("levels are top down", true);
  numLevels_ = conf.getInt("nlevels", 1);
  numLevels = numLevels_;
  halo_ = conf.getUnsigned("halo", 1);

  // A GSI-matching setup requires both the grid and the partitioner sub-configs (or neither).
  const bool hasGsiGrid = conf.has(GsiGridKey);
  const bool hasGsiPartitioner = conf.has(GsiPartitionerKey);
  if (!hasGsiGrid || !hasGsiPartitioner) {
    throw eckit::BadParameter(
        "GeometryAtlas: must specify GSI-matching grid AND partitioner, OR neither", Here());
  }
  // Reuse SABER's GSI-matching grid + south-to-north checkerboard partitioner.
  setupGsiMatchingGrid(conf, comm, grid_, functionSpace_, geomFields);
  // SABER returns an empty fieldset for the GSI path; add the "owned" mask that downstream
  // oops code expects.
  atlas::Field owned = functionSpace_.createField<int>(atlas::option::name("owned") |
                                                       atlas::option::levels(1));
  auto ownedView = atlas::array::make_view<int, 2>(owned);
  auto ghostView = atlas::array::make_view<int, 1>(functionSpace_.ghost());
  for (atlas::idx_t i = 0; i < ghostView.shape(0); ++i) {
    ownedView(i, 0) = (ghostView(i) > 0 ? 0 : 1);
  }
  geomFields.add(owned);
  functionSpace = functionSpace_;

  oops::Log::trace() << "GeometryAtlas constructor done" << std::endl;
}

// -----------------------------------------------------------------------------------------------

void GeometryGsibec::print(std::ostream &os) const {
  std::string prefix;
  os << "GSIBEC geometry grid:" << std::endl;
  if (grid_)
  {
    os << "- name: " << grid_.name() << std::endl;
    os << "- size: " << grid_.size() << std::endl;
  }
  if (partitioner_)
  {
    os << "Partitioner:" << std::endl;
    os << "- type: " << partitioner_.type() << std::endl;
  }
  os << "Function space:" << std::endl;
  os << "- type: " << functionSpace_.type() << std::endl;
  os << "- halo: " << halo_ << std::endl;
  os << "- levels: " << numLevels_ << std::endl;
}

// -----------------------------------------------------------------------------------------------

std::vector<double> GeometryGsibec::verticalCoord(std::string & /*vcUnits*/) const {
  std::stringstream errorMsg;
  errorMsg << "GeometryAtlas::verticalCoord is not implemented" << std::endl;
  throw eckit::NotImplemented(errorMsg.str(), Here());
}

// -----------------------------------------------------------------------------------------------

}  // namespace ijedi
