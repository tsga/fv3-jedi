#include <algorithm>
#include <memory>
#include <string>

#include "eckit/config/Configuration.h"
#include "eckit/config/LocalConfiguration.h"
#include "eckit/exception/Exceptions.h"

/*#include "ijedi/Geometry/base/GeometryBase.h"
#include "ijedi/Geometry/atlas/GeometryAtlas.h"
#include "ijedi/Geometry/fv3/GeometryFV3.h"
#include "ijedi/Geometry/gsibec/GeometryGsibec.h"
#include "ijedi/Geometry/mpas/GeometryMPAS.h"
#include "ijedi/Geometry/mom6/GeometryMOM6.h"*/

#include "fv3jedi/Geometry/gsibec/GeometryGsibec.h"
#include "fv3jedi/Geometry/base/GeometryBase.h"

namespace fv3jedi
{
  std::shared_ptr<GeometryBase> GeometryBase::create(const eckit::Configuration &geomConf,
                                                     const eckit::mpi::Comm &comm,
                                                     eckit::LocalConfiguration &geomVars,
                                                     atlas::FunctionSpace &functionSpace,
                                                     atlas::FieldSet &fieldSet,
                                                     bool &levelsAreTopDown, int &numLevels)
  {
    // Get the type
    std::string type;
    type = geomConf.getString("geometry_type");

  /*  if (type == "fv3")
    {
      return std::make_shared<GeometryFV3>(geomConf, comm, geomVars, functionSpace, fieldSet,
                                           levelsAreTopDown, numLevels);
    }
    if (type == "mpas")
    {
      return std::make_shared<GeometryMPAS>(geomConf, comm, geomVars, functionSpace, fieldSet,
                                            levelsAreTopDown, numLevels);
    }
    if (type == "mom6")
    {
      return std::make_shared<GeometryMOM6>(geomConf, comm, geomVars, functionSpace, fieldSet,
                                            levelsAreTopDown, numLevels);
    }
    if (type == "atlas")
    {
      return std::make_shared<GeometryAtlas>(geomConf, comm, geomVars, functionSpace, fieldSet,
                                             levelsAreTopDown, numLevels);
    }*/
    if (type == "gsibec")
    {
      return std::make_shared<GeometryGsibec>(geomConf, comm, geomVars, functionSpace, fieldSet,
                                              levelsAreTopDown, numLevels);
    }

    throw eckit::BadValue("Unsupported geometry type: " + type,
                          Here());
  }

}  // namespace fv3jedi
