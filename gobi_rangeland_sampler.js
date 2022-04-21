var copernicus_collection = ee.ImageCollection('COPERNICUS/S2_SR')
  .filter(ee.Filter.bounds(sample_points));
var modis_collection = ee.ImageCollection('MODIS/006/MOD11A2')
  .filter(ee.Filter.bounds(sample_points));
var chirps_collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
  .filter(ee.Filter.bounds(sample_points));

function maskCloudAndShadows(image) {
  var cloudProb = image.select('MSK_CLDPRB');
  var snowProb = image.select('MSK_SNWPRB');
  var cloud = cloudProb.lt(5);
  var snow = snowProb.lt(5);
  var scl = image.select('SCL');
  var shadow = scl.eq(3); // 3 = cloud shadow
  var cirrus = scl.eq(10); // 10 = cirrus
  // Cloud probability less than 5% or cloud shadow classification
  var mask = (cloud.and(snow)).and(cirrus.neq(1)).and(shadow.neq(1));
  return image.updateMask(mask);
}

function calculateNDVI(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi');
  return ndvi;
}

var sample_array = [];
for (var year=2019; year <= 2021; year++) {
  var image_collection = null;
  var local_points = sample_points
    .filter(ee.Filter.eq('Year', year))
    .select(['Condition', 'Veg_cover']);
  var ndvi_dates = [year+'-04-01', year+'-04-30', year+'-05-01', year+'-05-30', year+'-06-01', year+'-06-30', year+'-07-01', year+'-07-31', year+'-08-01', year+'-08-31', year+'-09-01', year+'-09-30', year+'-10-01', year+'-10-31'];
  for (var month_index = 0; month_index < ndvi_dates.length/2; month_index++){
    var startDate =  ndvi_dates[month_index*2];
    var endDate = ndvi_dates[month_index*2+1];
    var ndvi_fieldname = 'ndvi_'+startDate.slice(5,10);

    var ndvi = copernicus_collection
      .filterDate(startDate, endDate)
      .map(maskCloudAndShadows)
      .map(calculateNDVI)
      .mean()
      .select(['ndvi'], [ndvi_fieldname]);
    if (image_collection === null) {
      image_collection = ndvi;
    } else {
      image_collection = image_collection.addBands(ndvi);
    }
  }

  var year_dates = [year+'-01-01', year+'-01-31', year+'-02-01', year+'-02-28', year+'-03-01', year+'-03-31', year+'-04-01', year+'-04-30', year+'-05-01', year+'-05-31', year+'-06-01', year+'-06-30', year+'-07-01', year+'-07-31', year+'-08-01', year+'-08-31', year+'-09-01', year+'-09-30', year+'-10-01', year+'-10-31', year+'-11-01', year+'-11-30', year+'-12-01', year+'-12-31'];
  for (var month_index = 0; month_index < year_dates.length/2; month_index++){
    var startDate =  year_dates[month_index*2];
    var endDate = year_dates[month_index*2+1];

    var fieldname = startDate.slice(5,10);

    var chirps = chirps_collection
      .filterDate(startDate, endDate)
      .mean()
      .select(['precipitation'], ['precipitation_'+fieldname]);
    image_collection = image_collection.addBands(chirps);

    var modis = modis_collection
      .filter(ee.Filter.date(startDate, endDate))
      .mean()
      .select(['LST_Day_1km', 'LST_Night_1km'], ['LST_Day_1km_'+fieldname, 'LST_Night_1km_'+fieldname]);
    image_collection = image_collection.addBands(modis);
  }
  sample_array.push(image_collection.sampleRegions({
    collection: local_points,
    scale: .1,
    geometries: true,
  }));
}

var total_samples = ee.FeatureCollection(sample_array).flatten();

print(total_samples);

Map.addLayer(total_samples, null, 'Sample points');
Map.centerObject(total_samples);

Export.table.toDrive({
  collection: total_samples,
  folder: 'gee_export',
  fileNamePrefix: 'gobi_rangeland_ndvi_monthly'
});
