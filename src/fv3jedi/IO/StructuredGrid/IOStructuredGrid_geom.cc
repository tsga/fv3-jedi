/*
 * (C) Copyright 2025- UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include <netcdf.h>

#include <map>
#include <vector>

#include "atlas/functionspace.h"
#include "atlas/grid.h"

#include "oops/base/GeometryData.h"
#include "oops/util/FieldSetHelpers.h"
#include "oops/util/Logger.h"
#include "oops/util/stringFunctions.h"
#include "oops/util/Timer.h"

#include "fv3jedi/FieldMetadata/FieldsMetadata.h"
#include "fv3jedi/Geometry/Geometry.h"
#include "fv3jedi/Increment/Increment.h"
#include "fv3jedi/IO/StructuredGrid/IOStructuredGrid.h"
#include "fv3jedi/State/State.h"
#include "fv3jedi/Utilities/fv3jedi_vertical_remap.h"

namespace fv3jedi {
// -------------------------------------------------------------------------------------------------
static IOMaker<IOStructuredGrid> makerIOStructuredGrid_("structured grid");
static IOMaker<IOStructuredGrid> makerIOAuxGrid_("auxgrid");
// -------------------------------------------------------------------------------------------------
static inline void nc_rc(const int return_code, const std::string & operation) {
  if (return_code != NC_NOERR) {
    ABORT("IOStructuredGrid netCDF operation \'" + operation + "\' failed with error: "
          + nc_strerror(return_code));
  }
}

IOStructuredGrid::IOStructuredGrid(const Geometry & geom, const Parameters_ & params) 
  : IOBase(geom, params.toConfiguration()), geom_(geom), params_(params) {
    
    // 1. All ranks participate natively in the parallel layout built during initialization
    // 2. No `writeFunctionSpace_` or `interpolator_` members needed!
    oops::Log::trace() << classname() << " Constructor configured for pure geographic space." << std::endl;
}

void IOStructuredGrid::read(State & x, const eckit::LocalConfiguration & fileionames,
                            const eckit::LocalConfiguration & fileioscaling) const {
  util::Timer timer(classname(), "read state");

  // 1. Gather a local reference to the distributed geographic fieldset
  atlas::FieldSet fieldsGeographic;
  x.toFieldSet(fieldsGeographic);

  // 2. Allocate a temporary serial FieldSet for Rank 0 disk access
  // (Assuming you instantiate a flat serial space container `serialSpace_` in your class)
  atlas::FieldSet fieldsSerial;
  if (geom_.getComm().rank() == 0) {
    for (const auto & field : fieldsGeographic) {
      atlas::Field fSerial = serialSpace_->createField<double>(
          atlas::option::name(field.name()) | atlas::option::levels(field.levels()));
      fieldsSerial.add(fSerial);
    }
  }

  // 3. Extract the target array profiles from configuration parameters
  const auto & filenamesOpt = params_.filenames.value();
  if (filenamesOpt != boost::none) {
    for (const auto & filename : filenamesOpt.value()) {
      if (geom_.getComm().rank() == 0) {
        this->readStructuredFields(params_.datapath.value() + "/" + filename, 
                                   fieldsSerial, x.validTime(), fileionames, fileioscaling);
      }
    }
  }

  // 4. Distribute data directly via Atlas's built-in parallel grid scatter 
  readFunctionSpace_->scatter(fieldsSerial, fieldsGeographic);

  // 5. Update State buffers directly without computing global interpolations
  x.fromFieldSet(fieldsGeographic);
}

IOStructuredGrid::IOStructuredGrid(const Geometry & geom, const Parameters_ & params)
  : IOBase(geom, params.toConfiguration()), interpolator_(), readInterpolator_(), geom_(geom),
    gridStr_(""), params_(params), writeFunctionSpace_(), readFunctionSpace_()  {
  util::Timer timer(classname(), "IOStructuredGrid");
  oops::Log::trace() << classname() << " constructor starting" << std::endl;

  // Create the Atlas structured grid
  // --------------------------------
  // Create the string to determine the grid name for Atlas
  std::string outputGridType = params.outputGridType.value();

  // Convert the legacy gridtype to what Atlas expects
  if (outputGridType == "latlon") {
    outputGridType = "L" + std::to_string(4*(geom.npx()-1)) + "x" +
                     std::to_string(2*(geom.npy()-1)+1);
  } else if (outputGridType == "gaussian") {
    // Find best matching Gaussian grid
    outputGridType = "F" + std::to_string(geom.npy()-1);
  }

  // Assert that grid begins with either L or F
  if (outputGridType[0] == 'L') {
    gridStr_ = "latlon";
  } else if (outputGridType[0] == 'F') {
    gridStr_ = "gaussian";
  } else {
    // This code is only tested with latlon and regular Gaussian grids. With other grids the code
    // may run but with resulting files containing incorrect or jumbled data.
    ABORT("IOStructuredGrid: outputGridType must begin with L (latlon) or F (regular gaussian). ");
  }

  // Generate the Atlas grid object
  const atlas::Grid grid(outputGridType);

  // Make a custom serial distribution where all points live on rank 0
  // -----------------------------------------------------------------
  std::vector<int> zeros(grid.size(), 0);
  const atlas::grid::Distribution dist(geom.getComm().size(), grid.size(), zeros.data());

  // Create the configuration for the interpolation and populate with the communicator name
  // --------------------------------------------------------------------------------------
  eckit::LocalConfiguration atlas_conf;
  atlas_conf.set("mpi_comm", geom.getComm().name());

  // Structured grid function space
  // ------------------------------
  writeFunctionSpace_.reset(new atlas::functionspace::StructuredColumns(grid, dist, atlas_conf));

  // Create a GeometryData object
  // ----------------------------
  oops::GeometryData geomData(geom.functionSpace(), geom.fields(), geom.levelsAreTopDown(),
                              geom.getComm());
  
  // Create a generic interpolator for converting to the structured grid
  // -------------------------------------------------------------------
  interpolator_.reset(new oops::GlobalInterpolator(params.toConfiguration(), 
                                                  geomData,
                                                  *writeFunctionSpace_,
                                                  geom.getComm()));
  
  // -------------------------
  // READ mode: build INPUT geometry from file
  // -------------------------
  if (params_.geomfilename.value() == boost::none) {
    //ABORT("IOStructuredGrid: mode is 'read' but no input filename was specified. "
    //      "The 'input filename' option is required for read mode.");
    oops::Log::trace() << "IOStructuredGrid: no geomfilename was specified. "
          "The 'geomfilename' option is required for read mode. Program crashes if this is read mode." << std::endl;
  }
  else {

    const std::string inFile = *params_.geomfilename.value();

    size_t nLat = 0, nLon = 0;
    std::vector<double> file_lats;
    std::vector<double> file_lons;
    if (geom_.getComm().rank() == 0) {
      int ncid;
      nc_rc(nc_open(inFile.c_str(), NC_NOWRITE, &ncid), "nc_open " + inFile);

      const std::string latName = params_.latName.value();
      const std::string lonName = params_.lonName.value();
      const std::string latvarName = params_.latvarName.value();
      const std::string lonvarName = params_.lonvarName.value();

      int dim_lat, dim_lon;
      nc_rc(nc_inq_dimid(ncid, latName.c_str(), &dim_lat), "nc_inq_dimid " + latName);
      nc_rc(nc_inq_dimlen(ncid, dim_lat, &nLat), "nc_inq_dimlen " + latName);
      nc_rc(nc_inq_dimid(ncid, lonName.c_str(), &dim_lon), "nc_inq_dimid " + lonName);
      nc_rc(nc_inq_dimlen(ncid, dim_lon, &nLon), "nc_inq_dimlen " + lonName);

      file_lats.resize(nLat*nLon);
      file_lons.resize(nLon*nLat);

      int var_lat, var_lon;
      nc_rc(nc_inq_varid(ncid, latvarName.c_str(), &var_lat), "nc_inq_varid " + latvarName);
      nc_rc(nc_get_var_double(ncid, var_lat, file_lats.data()), "nc_get_var_double " + latvarName);

      nc_rc(nc_inq_varid(ncid, lonvarName.c_str(), &var_lon), "nc_inq_varid " + lonvarName);
      nc_rc(nc_get_var_double(ncid, var_lon, file_lons.data()), "nc_get_var_double " + lonvarName);

      nc_rc(nc_close(ncid), "nc_close");

      oops::Log::info() << "Input file grid: nLon=" << nLon << ", nLat=" << nLat << std::endl;
    }

    // Broadcast dims and coordinates
    geom_.getComm().broadcast(nLat, 0);
    geom_.getComm().broadcast(nLon, 0);
    oops::Log::info() << "1 Rank "<< geom_.getComm().rank() << std::endl;
    if (geom_.getComm().rank() != 0) {
      file_lats.resize(nLat*nLon);
      file_lons.resize(nLat*nLon);
    }
    oops::Log::info() << "2 Rank "<< geom_.getComm().rank() << std::endl;
    geom_.getComm().broadcast(file_lats, 0);
    geom_.getComm().broadcast(file_lons, 0);
    oops::Log::info() << "3 Rank "<< geom_.getComm().rank() << std::endl;

    // Use an Atlas L-grid only as the structured indexing/distribution container.
    // In HAFS regional applications, the input is treated as a file-defined
    // structured grid using explicit latitude/longitude coordinates from the input file.
    // The exact source coordinates are supplied below through the lonlat field.
    // Native F-grid support can be added later if needed.
    const std::string inputGridType = "L" + std::to_string(nLon) + "x" + std::to_string(nLat);
    const atlas::Grid inputGrid(inputGridType);
    oops::Log::info() << "4 Rank "<< geom_.getComm().rank() << std::endl;

    // -----------------------------------------------------------------------------
    // Build distributed structured FunctionSpace for the global file grid
    // -----------------------------------------------------------------------------
    atlas::grid::Partitioner partitioner("equal_regions");

    // --- IMPORTANT: enable halo/ghosts so haloExchange is real and seam stencils exist ---
    atlas_conf.set("halo", 2);
    oops::Log::info() << "5 Rank "<< geom_.getComm().rank() << std::endl;
    readFunctionSpace_.reset(new atlas::functionspace::StructuredColumns(inputGrid, partitioner, atlas_conf));
    oops::Log::info() << "6 Rank "<< geom_.getComm().rank() << std::endl;
    // --- Override lonlat using the configured input longitude/latitude coordinates ---
    atlas::FieldSet structuredAux;

    atlas::Field lonlat = readFunctionSpace_->createField<double>(
        atlas::option::name("lonlat") | atlas::option::variables(2));

    auto lonlatView = atlas::array::make_view<double,2>(lonlat);
    auto gidxView   = atlas::array::make_view<atlas::gidx_t,1>(readFunctionSpace_->global_index());
    auto ghostView  = atlas::array::make_view<int,1>(readFunctionSpace_->ghost());

    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (atlas::idx_t p = 0; p < lonlatView.shape(0); ++p) {
      lonlatView(p,0) = nan;
      lonlatView(p,1) = nan;
    }
    oops::Log::info() << "7 Rank "<< geom_.getComm().rank() << std::endl;
    // The configured latitude coordinate is stored in the same row order as the NetCDF variables.
    // Therefore, the structured row index j is mapped directly to file_lats[j].
    // Do not flip north-to-south files here. Latitude orientation is handled
    // only when computing optional regional subset indices.
    for (atlas::idx_t p = 0; p < lonlatView.shape(0); ++p) {
      if (ghostView(p) != 0) continue;              // owned only
      const auto gidx = gidxView(p);
      if (gidx <= 0) continue;
      const std::size_t g = static_cast<std::size_t>(gidx) - 1;
      const std::size_t i = g % nLon;
      const std::size_t j = g / nLon;
      lonlatView(p,0) = file_lons[i];
      lonlatView(p,1) = file_lats[j];
    }
    oops::Log::info() << "8 Rank "<< geom_.getComm().rank() << std::endl;
    // populate ghost lonlat consistently
    readFunctionSpace_->haloExchange(lonlat);

    structuredAux.add(lonlat);

    oops::GeometryData structuredGeomData(*readFunctionSpace_,
                                          structuredAux,
                                          geom.levelsAreTopDown(),
                                          geom.getComm());
    oops::Log::info() << "9 Rank "<< geom_.getComm().rank() << std::endl;
    // Reverse interpolator (structured -> model)
    eckit::LocalConfiguration interpConfig = params.toConfiguration();
    if (!interpConfig.has("local interpolator type")) {
      interpConfig.set("local interpolator type", "oops unstructured grid interpolator");
    }
    oops::Log::info() << "10 Rank "<< geom_.getComm().rank() << std::endl;
    // Only create interpolator once
    if (!readInterpolator_) {
      readInterpolator_.reset(new oops::GlobalInterpolator(interpConfig,
                                                          structuredGeomData,
                                                          geom.functionSpace(),
                                                          geom.getComm()));
      if (geom_.getComm().rank() == 0) {
        oops::Log::info() << "Interpolator created (will be reused for all files)" << std::endl;
      }
    }    
  }

  oops::Log::trace() << classname() << " constructor done" << std::endl;
}
// -------------------------------------------------------------------------------------------------
IOStructuredGrid::~IOStructuredGrid() {
  util::Timer timer(classname(), "~IOStructuredGrid");
  oops::Log::trace() << classname() << " destructor starting" << std::endl;
  oops::Log::trace() << classname() << " destructor done" << std::endl;
}

// -------------------------------------------------------------------------------------------------
void IOStructuredGrid::read(State & x, const eckit::LocalConfiguration & fileionames,
                            const eckit::LocalConfiguration & fileioscaling) const {
    util::Timer timer(classname(), "read state");
    oops::Log::trace() << classname() << " read state starting" << std::endl;

    // 1. Create a distributed model fieldset
    atlas::FieldSet fieldsCubeSphere;
    x.toFieldSet(fieldsCubeSphere);

    // 2. Create a temporary SERIAL FieldSet using writeFunctionSpace_ (allocated on Rank 0)
    atlas::FieldSet fieldsSerial;
    for (const auto & fieldCubeSphere : fieldsCubeSphere) {
      atlas::Field fieldSerial = writeFunctionSpace_->createField<double>(
          atlas::option::name(fieldCubeSphere.name()) | 
          atlas::option::levels(fieldCubeSphere.levels()));
      fieldsSerial.add(fieldSerial);
    }

    // 3. Retrieve the boost::optional vector from the parameter
    const auto & filenamesOpt = params_.filenames.value();

    if (filenamesOpt != boost::none) {
      const std::vector<std::string> & files = filenamesOpt.value();

      for (const auto & filename : files) {
        oops::Log::info() << classname() << " reading from file: " << filename << std::endl;

        // 4. Read data from disk exclusively on Rank 0 into the serial structure
        if (geom_.getComm().rank() == 0) {
          this->readStructuredFields(params_.datapath.value() + "/" + filename, fieldsSerial, x.validTime(), fileionames, fileioscaling);
        }
      }
    } else {
      ABORT("IOStructuredGrid::read: 'filenames' parameter is missing or unconfigured.");
    }
    oops::Log::info() << classname() << " done reading files " << std::endl;

    // 5. Set up the target distributed structured grid using readFunctionSpace_
    atlas::FieldSet fieldsGeographic;
    for (const auto & fieldCubeSphere : fieldsCubeSphere) {
      atlas::Field fieldGeographic = readFunctionSpace_->createField<double>(
          atlas::option::name(fieldCubeSphere.name()) | 
          atlas::option::levels(fieldCubeSphere.levels()));
      fieldsGeographic.add(fieldGeographic);
    }

    // 5b. Safely distribute the data from the Serial FieldSet (Rank 0) to the Parallel FieldSet (All Ranks)
    // This utilizes Atlas's built-in global-to-local indexing mechanics via a Scatter operation
    readFunctionSpace_->scatter(fieldsSerial, fieldsGeographic);

    // 6. Set interpolation metadata
    oops::Log::info() << classname() << "setting interp type " << std::endl;
    auto set_interp_type = [&](atlas::FieldSet & fs, const std::string & val) {
      for (auto & f : fs) {
        if (!f.metadata().has("interp_type")) f.metadata().set("interp_type", val);
      }
    };
    set_interp_type(fieldsGeographic, "default");
    set_interp_type(fieldsCubeSphere, "default");  

    /*auto set_bottm_lev = [&](atlas::FieldSet & fs, const std::string & val) {
      for (auto & f : fs) {
        if (!f.metadata().has("nearest 3d level")) f.metadata().set("nearest 3d level", val);
      }
    };
    set_bottm_lev(fieldsGeographic, "bottom"); 
    set_bottm_lev(fieldsCubeSphere, "bottom"); */

    // 7. Interpolate safely from distributed geographic fields to CubeSphere
    oops::Log::info() << classname() << "interpolating geog to c3g " << std::endl;
    readInterpolator_->apply(fieldsGeographic, fieldsCubeSphere);

    oops::Log::info() << classname() << "populating states with fields " << std::endl;
    
    // 8. Populate the State object
    x.fromFieldSet(fieldsCubeSphere);

    oops::Log::trace() << classname() << " read state done" << std::endl;
}
  
// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::read(Increment & dx, const eckit::LocalConfiguration & fileionames,
                                const eckit::LocalConfiguration & fileioscaling) const {
  ABORT("IOStructuredGrid::read(Increment) not implemented");
}

// -------------------------------------------------------------------------------------------------

template <typename T>
void IOStructuredGrid::interpAndWrite(const T & obj, const std::string & label,
                                      const eckit::LocalConfiguration & fileionames,
                                      const eckit::LocalConfiguration & fileioscaling) const {
  util::Timer timer(classname(), "write " + label);
  oops::Log::trace() << classname() << " write " << label << " starting" << std::endl;

  // Create field sets
  atlas::FieldSet fieldsCubeSphere;
  atlas::FieldSet fieldsGeographic;
  obj.toFieldSet(fieldsCubeSphere);

  // Apply interpolation
  interpolator_->apply(fieldsCubeSphere, fieldsGeographic);

  // If orography filename is provided, remap vertical coordinates
  if ( !params_.doVerticalRemapping.value() ) {
    // Write to disk if rank 0
    if (geom_.getComm().rank() == 0) {
      this->writeStructuredFields(fieldsGeographic, obj.validTime(), fileionames, fileioscaling);
    }
  } else {
    ASSERT(params_.orographyFilename.value() != boost::none);

    // Define orography variables
    atlas::FieldSet fieldsOrog;
    atlas::Field zsOrogNew = fieldsGeographic["geopotential_height_at_surface"].clone();
    fieldsOrog.add(zsOrogNew);

    // Write to disk if rank 0
    if (geom_.getComm().rank() == 0) {
      // Read structured-grid orography from file
      const std::string orogFilename = params_.orographyFilename.value().value();
      this -> readStructuredFields(orogFilename, fieldsOrog, obj.validTime(),
                                   fileionames, fileioscaling);

      // Remap the vertical coordinates to account for orography changes
      fv3jedi::VertRemap vert_remap(geom_, fieldsOrog);
      atlas::FieldSet fieldsGeographicRemap = vert_remap.remap(fieldsGeographic);

      // Write to disk
      this->writeStructuredFields(fieldsGeographicRemap, obj.validTime(),
                                  fileionames, fileioscaling);
    }
  }

  oops::Log::trace() << classname() << " write " << label << " done" << std::endl;
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::write(const State & x, const eckit::LocalConfiguration & fileionames,
                                const eckit::LocalConfiguration & fileioscaling) const {
  this->interpAndWrite(x, "state", fileionames, fileioscaling);
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::write(const Increment & dx, const eckit::LocalConfiguration & fileionames,
                             const eckit::LocalConfiguration & fileioscaling) const {
  this->interpAndWrite(dx, "increment", fileionames, fileioscaling);
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::writeStructuredFields(const atlas::FieldSet & fields,
                                             const util::DateTime & time,
                                             const eckit::LocalConfiguration & ioNames,
                                             const eckit::LocalConfiguration & ioScaling) const {
  // NetCDF IDs
  // ----------
  int fileId;

  // Dimension indices
  int latId;
  int lonId;
  int levId;
  int edgId;
  int forId;
  int timId;

  // Variable indices
  int fIv;
  std::map<std::string, int> fieldIvs;

  // Get ak/bk for writing
  // ---------------------
  std::vector<double> ak = geom_.ak();
  std::vector<double> bk = geom_.bk();

  // Get the name of the file and adjust with datetime
  // -------------------------------------------------
  std::string pathFile = params_.filename.value();

  // For backward compatibility add some things to the filename if not already present
  if (pathFile.find("%Y") == std::string::npos) {
    pathFile += "%Y%m%d_%H%M%Sz";
  }
  if (pathFile.find(".nc") == std::string::npos) {
    pathFile += ".nc4";
  }

  // Format the datetime string
  pathFile = time.formatString(pathFile);

  // Replace member number (ensemble applciaitons)
  util::stringfunctions::swapNameMember(params_.toConfiguration(), pathFile);

  // Create a file to write fields into
  // ----------------------------------
  nc_rc(nc_create(pathFile.c_str(), NC_CLOBBER | NC_NETCDF4, &fileId), "nc_create" + pathFile);

  // Create regular grid for determining lat/lon values
  // --------------------------------------------------
  const atlas::RegularGrid regGrid(writeFunctionSpace_->grid());

  // Define the dimensions in the file
  // ---------------------------------
  const int nLat = regGrid.ny();
  const int nLon = regGrid.nx();
  const int nLev = geom_.npz();
  const int nEdg = geom_.npz() + 1;
  const int nFor = 4;
  const int nTim = 1;

  nc_rc(nc_def_dim(fileId, params_.latName.value().c_str(), nLat, &latId), "nc_def_dim (lat)");
  nc_rc(nc_def_dim(fileId, params_.lonName.value().c_str(), nLon, &lonId), "nc_def_dim (lon)");
  nc_rc(nc_def_dim(fileId, params_.levName.value().c_str(), nLev, &levId), "nc_def_dim (lev)");
  nc_rc(nc_def_dim(fileId, params_.edgName.value().c_str(), nEdg, &edgId), "nc_def_dim (edg)");
  nc_rc(nc_def_dim(fileId, params_.forName.value().c_str(), nFor, &forId), "nc_def_dim (for)");
  nc_rc(nc_def_dim(fileId, params_.timName.value().c_str(), nTim, &timId), "nc_def_dim (tim)");

  // Define the dimensions variables in the file
  // -------------------------------------------
  std::vector<double> latArr(nLat);
  std::vector<double> lonArr(nLon);
  std::vector<int> levArr(nLev);
  std::vector<int> edgArr(nEdg);
  std::vector<int> forArr(nFor);
  std::vector<int> timArr(nTim);

  for (int i = 0; i < nLat; ++i) {
    latArr[i] = regGrid.y(nLat - 1 - i);
  }
  for (int i = 0; i < nLon; ++i) {
    lonArr[i] = regGrid.x(i);
  }
  for (int i = 0; i < nLev; ++i) {
    levArr[i] = i + 1;
  }
  for (int i = 0; i < nEdg; ++i) {
    edgArr[i] = i + 1;
  }
  for (int i = 0; i < nFor; ++i) {
    forArr[i] = i + 1;
  }
  for (int i = 0; i < nTim; ++i) {
    timArr[i] = i + 1;
  }

  // Write the dimension variables (and attributes) to the file
  // ----------------------------------------------------------
  nc_rc(nc_def_var(fileId, params_.latName.value().c_str(), NC_DOUBLE, 1, &latId, &fIv),
        "nc_def_var (lat)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("degrees_north"), "degrees_north"),
        "nc_put_att_text (lat)");
  fieldIvs[params_.latName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.lonName.value().c_str(), NC_DOUBLE, 1, &lonId, &fIv),
        "nc_def_var (lon)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("degrees_east"), "degrees_east"),
        "nc_put_att_text (lon)");
  fieldIvs[params_.lonName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.levName.value().c_str(), NC_INT, 1, &levId, &fIv),
        "nc_def_var (lev)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"),
        "nc_put_att_text (lev)");
  fieldIvs[params_.levName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.edgName.value().c_str(), NC_INT, 1, &edgId, &fIv),
        "nc_def_var (edg)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"),
        "nc_put_att_text (edg)");
  fieldIvs[params_.edgName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.forName.value().c_str(), NC_INT, 1, &forId, &fIv),
        "nc_def_var (for)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"),
        "nc_put_att_text (for)");
  fieldIvs[params_.forName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.timName.value().c_str(), NC_INT, 1, &timId, &fIv),
        "nc_def_var (tim)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"),
        "nc_put_att_text (tim)");
  fieldIvs[params_.timName.value()] = fIv;

  // Define some categories of dimension IDs for fields
  // --------------------------------------------------
  std::map<int, std::vector<int>> fieldDims;
  fieldDims[nLev] = {timId, levId, latId, lonId};  // Fields at levels
  fieldDims[nEdg] = {timId, edgId, latId, lonId};  // Fields at edges
  fieldDims[4] = {timId, forId, latId, lonId};     // Fields at four levels
  fieldDims[1] = {timId, latId, lonId};            // Fields at surface
  fieldDims[0] = {timId, latId, lonId};            // Fields at surface

  // Set float precision for fields
  // ------------------------------
  const int floatPrecision = params_.floatPrecision.value();
  const int ncPrec = (floatPrecision == 4) ? NC_FLOAT : NC_DOUBLE;

  // Define all the fields that will be written
  // ------------------------------------------
  for (auto& field : fields) {
    // Get number of levels for this field
    const int nLevField = field.shape(1);

    // Get dimensions for this field from map
    auto it = fieldDims.find(nLevField);
    if (it == fieldDims.end()) {
      std::ostringstream oss;
      oss << "IOStructuredGrid::writeStructuredFields: "
          << "No entry in fieldDims for field '" << field.name()
          << "' with " << nLevField << " levels.";
      ABORT(oss.str());
    }
    const auto &dims = it->second;

    // Look for fieldname in the iofile configuration and use the value if key found
    const std::string fieldLong = field.name();
    const char * fieldLongC = fieldLong.c_str();

    // Get the fieldmetadata for this field
    const FieldMetadata & fieldMetadata = geom_.fieldsMetaData().getFieldMetadata(fieldLong);
    std::string unitsStr = fieldMetadata.getVarUnits();
    const char * units = unitsStr.c_str();

    std::string fieldName = fieldLong;
    if (ioNames.has(fieldName)) {
      fieldName = ioNames.getString(fieldLong);
    }

    // Define the field in the file
    nc_rc(nc_def_var(fileId, fieldName.c_str(), ncPrec, dims.size(), dims.data(), &fIv),
          "nc_def_var " + fieldName);
    nc_rc(nc_put_att_text(fileId, fIv, "units", strlen(units), units),
          "nc_put_att_text " + fieldName + " units");
    nc_rc(nc_put_att_text(fileId, fIv, "long_name", strlen(fieldLongC), fieldLongC),
          "nc_put_att_text " + fieldName + " long_name");

    // Insert field into the fieldIvs map
    fieldIvs[field.name()] = fIv;
  }

  // Write ak/bk to the file as global attributes
  // --------------------------------------------
  nc_rc(nc_put_att_double(fileId, NC_GLOBAL, "ak", NC_DOUBLE, ak.size(), ak.data()),
          "nc_put_att_double (ak)");
  nc_rc(nc_put_att_double(fileId, NC_GLOBAL, "bk", NC_DOUBLE, bk.size(), bk.data()),
          "nc_put_att_double (bk)");
  nc_rc(nc_put_att_text(fileId, NC_GLOBAL, "grid", strlen(gridStr_.c_str()), gridStr_.c_str()),
          "nc_put_att_text (grid)");
  nc_rc(nc_put_att_int(fileId, NC_GLOBAL, "im", NC_INT, 1, &nLon),
          "nc_put_att_int (im)");
  nc_rc(nc_put_att_int(fileId, NC_GLOBAL, "jm", NC_INT, 1, &nLat),
          "nc_put_att_int (im)");

  // End definition mode
  // -------------------
  nc_rc(nc_enddef(fileId), "nc_enddef");

  // Write coordinate data into the file
  // -----------------------------------
  nc_rc(nc_put_var_double(fileId, fieldIvs[params_.latName.value()], latArr.data()),
        "nc_put_var_double (lat)");
  nc_rc(nc_put_var_double(fileId, fieldIvs[params_.lonName.value()], lonArr.data()),
        "nc_put_var_double (lon)");
  nc_rc(nc_put_var_int(fileId, fieldIvs[params_.levName.value()], levArr.data()),
        "nc_put_var_int (lev)");
  nc_rc(nc_put_var_int(fileId, fieldIvs[params_.edgName.value()], edgArr.data()),
        "nc_put_var_int (edg)");
  nc_rc(nc_put_var_int(fileId, fieldIvs[params_.timName.value()], timArr.data()),
        "nc_put_var_int (tim)");

  // Write the fields into the file
  // ------------------------------
  for (auto& field : fields) {
    // Get number of levels for this field
    const int nLevField = field.shape(1);

    // Create a rank 2 view of the field
    const auto fieldView = atlas::array::make_view<double, 2>(field);

    // Vector to hold the packed field
    std::vector<double> values(nLat*nLon*nLevField);

    // Loop over dimensions and pack the field
    for (size_t k = 0; k < nLevField; ++k) {
      for (size_t j = 0; j < nLat; ++j) {
        for (size_t i = 0; i < nLon; ++i) {
          values[k*nLat*nLon + j*nLon + i] = fieldView((nLat - 1 - j) * nLon + i, k);
        }
      }
    }

    // Write the field to the file
    nc_rc(nc_put_var_double(fileId, fieldIvs[field.name()], values.data()),
          "nc_put_var_double " + field.name());
  }

  // Close netCDF file
  // -----------------
  nc_rc(nc_close(fileId), "nc_close");
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::readStructuredFields(const std::string pathFile,
                                            atlas::FieldSet & fields,
                                            const util::DateTime & time,
                                            const eckit::LocalConfiguration & ioNames,
                                            const eckit::LocalConfiguration & ioScaling) const {
  // NetCDF IDs
  int fileId;

  // Open a file to read fields from
  // -------------------------------
  oops::Log::info() << "Reading file " << pathFile << std::endl;
  nc_rc(nc_open(pathFile.c_str(), NC_NOWRITE, &fileId), "nc_open " + pathFile);

  // Get file number of dimensions + their IDs
  // -----------------------------------------
  int ndims;
  nc_rc(nc_inq_ndims(fileId, &ndims), "nc_inq_ndims");

  std::vector<int> dimids(ndims);
  nc_rc(nc_inq_dimids(fileId, &ndims, dimids.data(), 0), "nc_inq_dimids");

  // Create regular grid for determining lat/lon values
  // --------------------------------------------------
  const atlas::RegularGrid regGrid(writeFunctionSpace_->grid());

  // Get grid dimensions
  // -------------------
  const int nLat = regGrid.ny();
  const int nLon = regGrid.nx();
  const int nLev = geom_.npz();
  const int nEdg = geom_.npz() + 1;
  const int nTim = 1;

  // Ensure that the lat and lon dimensions are found and have the correct lengths
  size_t dimSize;
  bool hasLat = false;
  bool hasLon = false;
  bool hasLev = false;
  bool hasEdg = false;
  bool hasTim = false;
  int latId;
  int lonId;
  int levId;
  int edgId;
  int timId;
  for (int i = 0; i < ndims; ++i) {
    // Get the name and size of the dimension
    char dimName[NC_MAX_NAME + 1];
    nc_rc(nc_inq_dim(fileId, dimids[i], dimName, &dimSize), "nc_inq_dim");

    if (std::string(dimName) == params_.latName.value().c_str()) {
      hasLat = true;
      latId = dimids[i];
      ASSERT(dimSize == nLat);
    } else if (std::string(dimName) == params_.lonName.value().c_str()) {
      hasLon = true;
      lonId = dimids[i];
      ASSERT(dimSize == nLon);
    } else if (std::string(dimName) == params_.levName.value().c_str()) {
      hasLev = true;
      levId = dimids[i];
      ASSERT(dimSize == nLev);
    } else if (std::string(dimName) == params_.edgName.value().c_str()) {
      hasEdg = true;
      edgId = dimids[i];
      ASSERT(dimSize == nEdg);
    } else if (std::string(dimName) == params_.timName.value().c_str()) {
      hasTim = true;
      timId = dimids[i];
      ASSERT(dimSize == nTim);
    }
  }

  // Ensure required dimensions were found
  ASSERT(hasLat);
  ASSERT(hasLon);
  ASSERT(hasLev);
  ASSERT(hasEdg);
  ASSERT(hasTim);

  // Read the fields from the file
  // -------------------------------
  for (auto & field : fields) {
    // Get IO name for this field
    std::string fieldName = field.name();
    if (ioNames.has(fieldName)) {
      fieldName = ioNames.getString(field.name());
    }
    oops::Log::info() << "Field " << fieldName << std::endl;
    // Get the variable ID for this field
    int varId;
    int status = nc_inq_varid(fileId, fieldName.c_str(), &varId);
    if (status == NC_ENOTVAR) {
      // Variable is not in this file; skip it silently (or log an info message)
      oops::Log::info() << "Field " << fieldName << " not found in this file, skipping..." << std::endl;
      continue; 
    } else {
      // Check for any other unexpected NetCDF errors
      nc_rc(status, "nc_inq_varid " + fieldName);
    }
    // nc_rc(nc_inq_varid(fileId, fieldName.c_str(), &varId), "nc_inq_varid " + fieldName);


    // Get number of dimensions + their IDs
    int ndims;
    int dimids[NC_MAX_VAR_DIMS];
    nc_rc(nc_inq_var(fileId, varId,
                     nullptr,   // var name (unused)
                     nullptr,   // type (unused)
                     &ndims,
                     dimids,
                     nullptr),  // attributes (unused)
          "nc_inq_var");

    // Ensure that the field has either 3 or 4 dimensions
    ASSERT(ndims == 3 || ndims == 4);

    // Ensure that the dimensions are in the expected order
    size_t nLevField = 0;
    if ( ndims == 3 ) {
      ASSERT(dimids[0] == timId &&
             dimids[1] == latId &&
             dimids[2] == lonId);
      nLevField = 1;
    } else if ( ndims == 4 ) {
      ASSERT((dimids[0] == timId &&
              dimids[1] == levId &&
              dimids[2] == latId &&
              dimids[3] == lonId) ||
             (dimids[0] == timId &&
              dimids[1] == edgId &&
              dimids[2] == latId &&
              dimids[3] == lonId));

      nLevField = field.shape(1);
      if ( dimids[1] == edgId ) {
        ASSERT(nLevField == nEdg);
      } else {
        ASSERT(nLevField == nLev);
      }
    }

    // Read the variable data
    std::vector<double> values(field.size());

    nc_rc(nc_get_var_double(fileId, varId, values.data()), "nc_get_var_double " + fieldName);

    // Create field and unpack data into it
    auto fieldView = atlas::array::make_view<double, 2>(field);

    // Corrected unpacking loop in readStructuredFields:
    for (size_t k = 0; k < nLevField; ++k) {
      for (size_t j = 0; j < nLat; ++j) {
        for (size_t i = 0; i < nLon; ++i) {
          // REMOVED "(nLat - 1 - j)" to match the constructor layout
          // fieldView((nLat - 1 - j) * nLon + i, k) = values[ k*nLat*nLon + j*nLon + i ];
          fieldView(j * nLon + i, k) = values[ k*nLat*nLon + j*nLon + i ];
        }
      }
    }
  }
  // Close file
  nc_rc(nc_close(fileId), "nc_close");
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::print(std::ostream & os) const {
  os << classname() << " IO using Atlas Structured Grid";
}

// -------------------------------------------------------------------------------------------------

}  // namespace fv3jed
