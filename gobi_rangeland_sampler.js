sample_points = sample_points.select('Rangelandmetricscore');

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

var image_collection = null;

var year = 2019;
var ndvi_dates = ['2019-07-01', '2019-07-15', '2019-07-16', '2019-07-31', '2019-08-01', '2019-08-15', '2019-08-16', '2019-08-31', '2019-09-01', '2019-09-15', '2019-09-16', '2019-09-30', '2019-10-01', '2019-10-15', '2019-10-16', '2019-10-31'];

for (var month_index = 0; month_index < ndvi_dates.length/2; month_index++){
  var startDate =  ndvi_dates[month_index*2];
  var endDate = ndvi_dates[month_index*2+1];
  var ndvi_fieldname = 'ndvi_'+startDate;

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

var year_dates = ['2019-01-01', '2019-01-31', '2019-02-01', '2019-02-28', '2019-03-01', '2019-03-31', '2019-04-01', '2019-04-30', '2019-05-01', '2019-05-31', '2019-06-01', '2019-06-30', '2019-07-01', '2019-07-31', '2019-08-01', '2019-08-31', '2019-09-01', '2019-09-30', '2019-10-01', '2019-10-31', '2019-11-01', '2019-11-30', '2019-12-01', '2019-12-31'];
for (var month_index = 0; month_index < year_dates.length/2; month_index++){
  var startDate =  year_dates[month_index*2];
  var endDate = year_dates[month_index*2+1];

  var chirps = chirps_collection
      .filterDate(startDate, endDate)
      .mean()
      .select(['precipitation'], ['precipitation_'+startDate]);
  image_collection = image_collection.addBands(chirps);

  var modis = modis_collection
    .filter(ee.Filter.date(startDate, endDate))
    .mean()
    .select(['LST_Day_1km', 'LST_Night_1km'], ['LST_Day_1km_'+startDate, 'LST_Night_1km_'+startDate]);
  image_collection = image_collection.addBands(modis);
}

var ndvi_samples = image_collection.sampleRegions({
  collection: sample_points,
  scale: .1,
});

print(ndvi_samples);
print(sample_points);

Map.addLayer(sample_points, null, 'Sample points');
Map.centerObject(sample_points);
