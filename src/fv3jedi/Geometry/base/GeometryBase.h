#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "eckit/config/LocalConfiguration.h"
#include "eckit/mpi/Comm.h"

#include "atlas/field.h"
#include "atlas/functionspace.h"

namespace eckit
{
  class Configuration;
}

namespace fv3jedi
{

  class GeometryBase
  {
   public:
    virtual ~GeometryBase() = default;

    static std::shared_ptr<GeometryBase> create(const eckit::Configuration &,
                                                const eckit::mpi::Comm &,
                                                eckit::LocalConfiguration &,
                                                atlas::FunctionSpace &,
                                                atlas::FieldSet &,
                                                bool &, int &);
    virtual void print(std::ostream &) const = 0;
    virtual std::vector<double> verticalCoord(std::string &) const = 0;
  };

}  // namespace fv3jedi
