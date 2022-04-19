
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

var year = 2019;
var copernicus = ee.ImageCollection('COPERNICUS/S2_SR');
var dates = ['2019-01-01', '2019-01-31', '2019-02-01', '2019-02-28', '2019-03-01', '2019-03-31', '2019-04-01', '2019-04-30', '2019-05-01', '2019-05-31', '2019-06-01', '2019-06-30', '2019-07-01', '2019-07-31', '2019-08-01', '2019-08-31', '2019-09-01', '2019-09-30', '2019-10-01', '2019-10-31', '2019-11-01', '2019-11-30', '2019-12-01', '2019-12-31'];
var ndvi_array = [];
var ndvi_fieldname_array = [];
var ndvi_image = null;
sample_points = sample_points.select('Rangelandmetricscore');
for (var month_index = 0; month_index < 12; month_index++){
  var startDate =  dates[month_index*2];
  var endDate = dates[month_index*2+1];
  var ndvi_fieldname = 'ndvi_2019_'+(1+month_index);
  ndvi_fieldname_array.push(ndvi_fieldname);

  var ndvi = copernicus
      .filterDate(startDate, endDate)
      .map(maskCloudAndShadows)
      .map(calculateNDVI)
      .filter(ee.Filter.bounds(sample_points)).mean().select(['ndvi'], [ndvi_fieldname]);
  if (ndvi_image === null) {
    ndvi_image = ndvi;
  } else {
    ndvi_image = ndvi_image.addBands(ndvi);
  }
}

print(ndvi_image);

var ndvi_samples = ndvi_image.sampleRegions({
  collection: sample_points,
  scale: 10,
});

print(ndvi_samples);

/*
var ndvi_collection = ee.ImageColletion(ndvi_array);
print(ndvi_collection);

var ndvi_sample = ndvi_collection.reduce({
  collection: sample_points,
  reducer: ee.Reducer.first(),
  scale: 10});
print(ndvi_sample);

*/
  /*
  var ndvi_image = ndvi.map(function(image) {
  return image.select(ndvi_year).reduceRegions({
    collection: sample_points,
    reducer: ee.Reducer.first().setOutputs([ndvi_year]),
    scale: 10,
  })// reduceRegion doesn't return any output if the image doesn't intersect
    // with the point or if the image is masked out due to cloud
    // If there was no ndvi value found, we set the ndvi to a NoData value -9999
    .map(function(feature) {
    var ndvi = ee.List([feature.get(ndvi_year), -9999])
      .reduce(ee.Reducer.firstNonNull())
    return feature.set({ndvi_year: ndvi_year, 'imageID': image.id()})
    });
  }).flatten();
    */

var modis_collection = ee.ImageCollection('MODIS/006/MCD12Q2');
// these variables are measured in days since 1-1-1970
var julian_day_variables = [
    'Greenup_1',
    'MidGreenup_1',
    'Peak_1',
    'Maturity_1',
    'MidGreendown_1',
    'Senescence_1',
    'Dormancy_1',
    ];
// these variables are direct quantities
var raw_variables = [
    'EVI_Minimum_1',
    'EVI_Amplitude_1',
    'EVI_Area_1',
    'QA_Overall_1',
    ];

//Map.addLayer(all_bands, null, 'Image Layer');
Map.addLayer(sample_points, null, 'Sample points');
Map.centerObject(sample_points);
