// ===================================================
// PHASE 1: SENTINEL-2 NEIGHBORHOOD FEATURES (GEE)
// ===================================================

// ---------------------------------------------------
// 1. Load property table (CSV uploaded as Asset)
// ---------------------------------------------------
var properties = ee.FeatureCollection(
    "projects/asset_id"
  );
  
  // ---------------------------------------------------
  // 2. Sentinel-2 L2A collection
  // ---------------------------------------------------
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate("2022-01-01", "2024-01-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)); // relaxed to ensure coverage
  
  // ---------------------------------------------------
  // 3. Add spectral indices
  // ---------------------------------------------------
  function addIndices(image) {
    var ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI");
    var ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI");
    var ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI");
    return image.addBands([ndvi, ndbi, ndwi]);
  }
  
  var s2Indexed = s2.map(addIndices);
  
  // ---------------------------------------------------
  // 4. Feature extraction (FIXED GEOMETRY)
  // ---------------------------------------------------
  function extractFeatures(feature) {
  
    // 🔴 REBUILD POINT GEOMETRY FROM LAT / LONG
    var point = ee.Geometry.Point([
      ee.Number(feature.get("long")),
      ee.Number(feature.get("lat"))
    ]);
  
    // 500 m neighborhood
    var geom = point.buffer(250);
  
    var localCol = s2Indexed.filterBounds(geom);
  
    // Safe masked fallback
    var empty = ee.Image.constant(0)
      .rename("NDVI")
      .addBands(ee.Image.constant(0).rename("NDBI"))
      .addBands(ee.Image.constant(0).rename("NDWI"))
      .updateMask(ee.Image.constant(0));
  
    var image = ee.Image(
      ee.Algorithms.If(
        localCol.size().gt(0),
        localCol.median().select(["NDVI", "NDBI", "NDWI"]),
        empty
      )
    );
  
    // Mean
    var meanDict = image.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: geom,
      scale: 10,
      maxPixels: 1e9
    });
  
    // Max
    var maxDict = image.reduceRegion({
      reducer: ee.Reducer.max(),
      geometry: geom,
      scale: 10,
      maxPixels: 1e9
    });
  
    // Explicit property assignment
    return feature.set({
      ndvi_mean_500m: meanDict.get("NDVI"),
      ndbi_mean_500m: meanDict.get("NDBI"),
      ndwi_mean_500m: meanDict.get("NDWI"),
  
      ndvi_max_500m: maxDict.get("NDVI"),
      ndbi_max_500m: maxDict.get("NDBI"),
      ndwi_max_500m: maxDict.get("NDWI")
    });
  }
  
  // ---------------------------------------------------
  // 5. Apply extraction
  // ---------------------------------------------------
  var propertiesWithIndices = properties.map(extractFeatures);
  
  // ---------------------------------------------------
  // 6. Debug check (RUN THIS)
  // ---------------------------------------------------
  print('Sample feature:');
  
  // ---------------------------------------------------
  // 7. Export to CSV
  // ---------------------------------------------------
  Export.table.toDrive({
    collection: propertiesWithIndices,
    description: "sentinel_neighborhood_features",
    fileFormat: "CSV"
  });