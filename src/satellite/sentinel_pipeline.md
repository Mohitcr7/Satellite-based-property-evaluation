# Satellite-2 Extraction Pipeline

## 🌍 Overview

This guide walks you through extracting Sentinel-2 satellite features (NDVI, NDBI, NDWI) for property neighborhoods using **Google Earth Engine**.

---

## 📋 Prerequisites

### 1. Google Earth Engine Account (Free)

```bash
# Sign up at https://earthengine.google.com/
# Request access (usually approved within 1-2 hours)
```

### 2. Authenticate Earth Engine

```bash
pip install earthengine-api

earthengine authenticate

# Browser opens → authorize → copy auth key
```

### 3. Property Coordinates

Ensure your property data has:
- `id` (unique identifier)
- `latitude` (decimal degrees)
- `longitude` (decimal degrees)
- `date` (ISO format for time window) - optional

---

## 🛰️ Sentinel-2 Indices Explained

### NDVI (Normalized Difference Vegetation Index)

**Formula**: `(NIR - RED) / (NIR + RED)`

**Bands Used**: B8 (NIR, 10m), B4 (RED, 10m)

**Interpretation**:
- `-1.0 to 0`: No vegetation (water, urban)
- `0 to 0.3`: Sparse vegetation (shrub, grass)
- `0.3 to 0.6`: Moderate vegetation (mixed areas)
- `0.6 to 1.0`: Dense vegetation (forests, parks)

**Price Impact**: Higher NDVI = greener neighborhoods = higher property values (+3-5%)

### NDBI (Normalized Difference Built-up Index)

**Formula**: `(SWIR - NIR) / (SWIR + NIR)`

**Bands Used**: B11 (SWIR, 20m), B8 (NIR, 10m)

**Interpretation**:
- `< 0`: Vegetation-dominated
- `0 to 0.2`: Mixed urban-vegetation
- `> 0.2`: Built-up areas (buildings, roads)

**Price Impact**: Moderate NDBI = balanced development = premium; High NDBI = urban congestion = discount (-2-4%)

### NDWI (Normalized Difference Water Index)

**Formula**: `(GREEN - SWIR) / (GREEN + SWIR)`

**Bands Used**: B3 (GREEN, 10m), B11 (SWIR, 20m)

**Interpretation**:
- `< 0`: No water
- `0 to 0.3`: Low water presence
- `> 0.3`: Water bodies (lakes, rivers)

**Price Impact**: Proximity to water = scenic views = premium (+2-3%)

---

## 🔧 JavaScript Code for Google Earth Engine

Save as `gee_sentinel_extractor.js` and run in [Earth Engine Code Editor](https://code.earthengine.google.com/):

```javascript
// ============================================================================
// Sentinel-2 Neighborhood Feature Extractor
// Extract NDVI, NDBI, NDWI for property neighborhoods
// ============================================================================

// ============================================================================
// 1. LOAD AND FILTER SENTINEL-2 DATA
// ============================================================================

// Define your study area (example: Washington State)
var studyArea = ee.Geometry.Rectangle([-124.7, 45.5, -116.9, 49.1]);

// Define date range
var startDate = '2014-06-01';
var endDate = '2014-10-01';

// Load Sentinel-2 imagery
var sentinel2 = ee.ImageCollection('COPERNICUS/S2')
  .filterBounds(studyArea)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  // < 20% cloud
  .median();  // Composite all images → median value

print('Sentinel-2 Composite:', sentinel2);

// ============================================================================
// 2. COMPUTE SPECTRAL INDICES
// ============================================================================

// NDVI = (B8 - B4) / (B8 + B4)
var ndvi = sentinel2.normalizedDifference(['B8', 'B4']).rename('NDVI');

// NDBI = (B11 - B8) / (B11 + B8)
var ndbi = sentinel2.normalizedDifference(['B11', 'B8']).rename('NDBI');

// NDWI = (B3 - B11) / (B3 + B11)
var ndwi = sentinel2.normalizedDifference(['B3', 'B11']).rename('NDWI');

// Combine indices
var indices = ndvi.addBands(ndbi).addBands(ndwi);

print('Indices:', indices);

// ============================================================================
// 3. DEFINE PROPERTY LOCATIONS (Example - manually coded)
// ============================================================================

// Load your property CSV as a table (via Earth Engine interface)
// For now, using sample coordinates:
var properties = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([-122.3, 47.6]), {'id': '1'}),  // Seattle
  ee.Feature(ee.Geometry.Point([-122.2, 47.5]), {'id': '2'}),
  ee.Feature(ee.Geometry.Point([-122.1, 47.4]), {'id': '3'}),
  // Add more coordinates...
]);

// ============================================================================
// 4. EXTRACT NEIGHBORHOOD STATISTICS (500m BUFFER)
// ============================================================================

// Function to extract statistics for each property
var extractNeighborhoodStats = function(feature) {
  var point = feature.geometry();
  var buffer = point.buffer(500);  // 500m neighborhood buffer
  
  var stats = indices.reduceRegion({
    reducer: ee.Reducer.mean().combine(ee.Reducer.max(), null, true),
    geometry: buffer,
    scale: 10,  // 10m pixel size
    maxPixels: 1e6
  });
  
  return feature.set(stats);
};

// Apply extraction
var propertiesWithStats = properties.map(extractNeighborhoodStats);

print('Properties with Stats:', propertiesWithStats);

// ============================================================================
// 5. EXPORT RESULTS TO GOOGLE DRIVE / CSV
// ============================================================================

// Export as CSV
Export.table.toDrive({
  collection: propertiesWithStats,
  description: 'sentinel_neighborhood_features',
  fileFormat: 'CSV',
  folder: 'EarthEngine'  // Google Drive folder
});

print('Export started. Check Earth Engine "Tasks" tab.');

// ============================================================================
// 6. VISUALIZATION (Optional)
// ============================================================================

var ndviVis = {
  min: 0, max: 1, palette: ['#d73027', '#fee08b', '#1a9850']
};

Map.centerObject(studyArea, 8);
Map.addLayer(ndvi, ndviVis, 'NDVI');
Map.addLayer(ndbi, {min: 0, max: 0.5, palette: ['white', 'black']}, 'NDBI');
```

---

## 🔄 Python Workflow: From Earth Engine to CSV

### Step 1: Prepare Property Coordinates

```python
import pandas as pd
import ee

# Load properties
properties = pd.read_csv('train_cleaned.csv')
print(properties[['id', 'latitude', 'longitude']].head())
```

### Step 2: Authenticate & Initialize

```python
import ee

ee.Authenticate()  # One-time only
ee.Initialize(project='my-gcp-project')  # Use your GCP project ID
```

### Step 3: Extract Features (Batch)

```python
import ee
import pandas as pd

def extract_sentinel_features(lat, lon, start_date='2014-06-01', end_date='2014-10-01', buffer_m=500):
    """
    Extract NDVI, NDBI, NDWI for a single property
    """
    
    # Create point
    point = ee.Geometry.Point([lon, lat])
    buffer = point.buffer(buffer_m)
    
    # Load Sentinel-2
    sentinel2 = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(buffer) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median()
    
    # Compute indices
    ndvi = sentinel2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndbi = sentinel2.normalizedDifference(['B11', 'B8']).rename('NDBI')
    ndwi = sentinel2.normalizedDifference(['B3', 'B11']).rename('NDWI')
    
    indices = ndvi.addBands(ndbi).addBands(ndwi)
    
    # Reduce region
    stats = indices.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.max(), null, True),
        geometry=buffer,
        scale=10,
        maxPixels=1e6
    ).getInfo()
    
    return stats

# Apply to all properties
results = []
for idx, row in properties.iterrows():
    try:
        stats = extract_sentinel_features(row['latitude'], row['longitude'])
        stats['id'] = row['id']
        results.append(stats)
        print(f"✓ Extracted features for property {row['id']}")
    except Exception as e:
        print(f"✗ Failed for property {row['id']}: {e}")

# Convert to DataFrame
sentinel_df = pd.DataFrame(results)
sentinel_df.to_csv('sentinel_neighborhood_features.csv', index=False)
print(f"\n✅ Saved {len(sentinel_df)} properties to sentinel_neighborhood_features.csv")
```

---

## 📊 Output Format

```csv
id,NDVI_mean,NDVI_max,NDBI_mean,NDBI_max,NDWI_mean,NDWI_max
1,0.42,0.65,0.08,0.22,0.15,0.38
2,0.38,0.60,0.12,0.25,0.18,0.42
3,0.45,0.70,0.05,0.18,0.12,0.35
...
```

---

## ⚠️ Troubleshooting

### Issue: "CLOUDY_PIXEL_PERCENTAGE" filter removes all images

**Solution**: Lower cloud threshold or expand date range

```javascript
.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))  // More lenient
```

### Issue: NaN values in output

**Causes**:
- Property outside image bounds
- All pixels masked (dense clouds)
- Water body (B11 band issues)

**Solution**: Inspect manually in Earth Engine visualization

### Issue: Slow extraction for 10,000+ properties

**Solution**: Use parallel requests or batch export via Earth Engine

```python
# Batch export (larger area)
export_region = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([lon, lat]), {'id': prop_id})
    for prop_id, lon, lat in zip(props['id'], props['longitude'], props['latitude'])
])

Export.table.toDrive({
    collection: export_region,
    description: 'batch_sentinel_extraction',
    fileFormat: 'CSV'
})
```

---

## 📚 References

- [Sentinel-2 Bands](https://custom-scripts.sentinel-hub.com/sentinel2-bands-explorer/)
- [Earth Engine Docs](https://developers.google.com/earth-engine/guides)
- [NDVI Interpretation](https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index)
- [NDBI & Urban Analysis](https://www.usgs.gov/faqs/what-normalized-difference-built-index-ndbi)

---

**Last Updated**: January 2026