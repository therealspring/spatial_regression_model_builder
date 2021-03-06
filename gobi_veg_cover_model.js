var global_image_dict = {};

var veg_cover_model = {
    'gobi_veg_cover_model': gobi_veg_cover_model,
    'gobi_condition_model': gobi_condition_model,
    'gobi_variance_veg_cover_model': gobi_variance_veg_cover_model,
    'gobi_variance_condition_model': gobi_variance_condition_model,
    'gobi_veg_cover_ndvi_only_model': gobi_veg_cover_ndvi_only_model,
};

var property_by_model = {
  'gobi_veg_cover_model': 'Veg_cover',
  'gobi_condition_model': 'Condition',
  'gobi_variance_veg_cover_model': 'Veg_cover',
  'gobi_variance_condition_model': 'Condition',
  'gobi_veg_cover_ndvi_only_model': 'Veg_cover',
};

// Load the carbon models
var image_map = {};

var global_model_dict = {};

var first_panel = ui.Panel({
    style: {
        border: '10px',
        position: "top-center",
}});

first_panel.add(ui.Label({
    value: 'LOADING GOBI RANGELAND MODEL UI....',
    style: {
        fontSize: '24px',
        fontWeight: 'bold',
    }}));
Map.add(first_panel);


var default_year = 2019;

var legend_styles = {
    'black_to_red': ['000000', '005aff', '43c8c8', 'fff700', 'ff0000'],
    'blue_to_green': ['440154', '414287', '218e8d', '5ac864', 'fde725'],
    'cividis': ['00204d', '414d6b', '7c7b78', 'b9ac70', 'ffea46'],
    'viridis': ['440154', '355e8d', '20928c', '70cf57', 'fde725'],
    'blues': ['f7fbff', 'c6dbef', '6baed6', '2171b5', '08306b'],
    'reds': ['fff5f0', 'fcbba1', 'fb6a4a', 'cb181d', '67000d'],
    'turbo': ['321543', '2eb4f2', 'affa37', 'f66c19', '7a0403'],
};
var default_legend_style = 'blue_to_green';

function changeColorScheme(key, active_context) {
    active_context.visParams.palette = legend_styles[key];
    active_context.build_legend_panel();
    active_context.updateVisParams();
}

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

var linkedMap = ui.Map();
Map.setCenter(0, 0, 3);

var model_count = Object.keys(veg_cover_model).length;

function build_images_by_year(year) {
  var copernicus_collection = ee.ImageCollection('COPERNICUS/S2_SR')
    .filter(ee.Filter.bounds(gobi_poly));
  var modis_collection = ee.ImageCollection('MODIS/006/MOD11A2')
    .filter(ee.Filter.bounds(gobi_poly));
  var chirps_collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .filter(ee.Filter.bounds(gobi_poly));

  var ndvi_dates = [year+'-04-01', year+'-04-30', year+'-05-01', year+'-05-30', year+'-06-01', year+'-06-30', year+'-07-01', year+'-07-31', year+'-08-01', year+'-08-31', year+'-09-01', year+'-09-30', year+'-10-01', year+'-10-31'];
  for (var month_index = 0; month_index < ndvi_dates.length/2; month_index++){
    var startDate, endDate, fieldname;
    startDate =  ndvi_dates[month_index*2];
    endDate = ndvi_dates[month_index*2+1];
    fieldname = startDate.slice(5,10).replace(/-/g, "_");

    var ndvi = copernicus_collection
      .filterDate(startDate, endDate)
      .map(maskCloudAndShadows)
      .map(calculateNDVI);

    var ndvi_mean = ndvi
        .mean();
    global_image_dict['ndvi_'+fieldname] = ndvi_mean.select(['ndvi'], ['B0']);

    var ndvi_var = ndvi
        .reduce(ee.Reducer.variance());
    global_image_dict['ndvi_variance_'+fieldname] = ndvi_var.select(['ndvi_variance'], ['B0']);
  }

  var year_dates = [year+'-01-01', year+'-01-31', year+'-02-01', year+'-02-28', year+'-03-01', year+'-03-31', year+'-04-01', year+'-04-30', year+'-05-01', year+'-05-31', year+'-06-01', year+'-06-30', year+'-07-01', year+'-07-31', year+'-08-01', year+'-08-31'];
  for (month_index = 0; month_index < year_dates.length/2; month_index++){
    startDate =  year_dates[month_index*2];
    endDate = year_dates[month_index*2+1];
    fieldname = startDate.slice(5,10).replace(/-/g, "_");

    var chirps = chirps_collection
      .filterDate(startDate, endDate);

    var precip_mean = chirps
        .mean();
    global_image_dict['precipitation_'+fieldname] = precip_mean.select(['precipitation'], ['B0']);

    var precip_var = chirps
        .reduce(ee.Reducer.variance());
    global_image_dict['precipitation_variance_'+fieldname] = precip_var.select(['precipitation_variance'], ['B0']);

    var modis = modis_collection
      .filter(ee.Filter.date(startDate, endDate));
    var modis_mean = modis
      .mean();
    global_image_dict['LST_Day_1km_'+fieldname] = modis_mean.select(['LST_Day_1km'], ['B0']);
    global_image_dict['LST_Night_1km_'+fieldname] = modis_mean.select(['LST_Night_1km'], ['B0']);

    var modis_var = modis
      .reduce(ee.Reducer.variance());
    global_image_dict['LST_Day_1km_variance_'+fieldname] = modis_var.select(['LST_Day_1km_variance'], ['B0']);
    global_image_dict['LST_Night_1km_variance_'+fieldname] = modis_var.select(['LST_Night_1km_variance'], ['B0']);
  }
}

function make_rangeland_model(model_id, term_list, year) {
  build_images_by_year(year);

  var sample_array = [];

  var i = null;
  // First term is the intercept, and renaming the band B0 because raw
  //image bands are also named that
  var rangeland_model_image = ee.Image(
      term_list[0].properties.coef).rename('B0');
  for (i = 1; i < term_list.length; i++) {
      var term = term_list[i].properties;
      // convert any - or . to _ so it can be evaluated
      var term_equation = term.id;
      var expression_str = (
          term.coef/term.scale+"*("+
          (term_equation).replace(/-/g, "_").replace(/\./g, "_")+
          "-"+term.mean+")");

      // add this term to the regression calculation
      var term_image = ee.Image().expression({
          expression: expression_str,
          map: global_image_dict
      });
      rangeland_model_image = rangeland_model_image.add(term_image);
  }
  global_model_dict[model_id] = rangeland_model_image.clip(gobi_poly);
  model_count -= 1;
  if (model_count == 0) {
      init_ui();
  }
  return global_model_dict[model_id];
}

var model_term_map = {};

Object.keys(veg_cover_model).forEach(function (model_id) {
    var table = veg_cover_model[model_id];
    var table_to_list = table.toList(table.size());

    //Create Carbon Regression Image based on table coefficients
    table_to_list.evaluate(function (term_list) {
        model_term_map[model_id] = term_list;
        make_rangeland_model(model_id, term_list, default_year);
    });
});

function init_ui() {
    Map.clear();
    var linker = ui.Map.Linker([ui.root.widgets().get(0), linkedMap]);
    // Create a SplitPanel which holds the linked maps side-by-side.
    var splitPanel = ui.SplitPanel({
        firstPanel: linker.get(0),
        secondPanel: linker.get(1),
        orientation: 'horizontal',
        wipe: true,
        style: {stretch: 'both'}
    });
    ui.root.widgets().reset([splitPanel]);
    var panel_list = [];

    var active_context_map = {};
    [[Map, 'left'], [linkedMap, 'right']].forEach(function(mapside, index) {
        var active_context = {
            'raster_type': null,
            'last_layer': null,
            'raster': null,
            'point_val': null,
            'last_point_layer': null,
            'map': mapside[0],
            'legend_panel': null,
            'visParams': null,
            'validation_layer': null,
            'validation_check': null,
            'active_map_layer_id': null,
            'gobi_poly_loaded': false,
            'chart_panel': null,
            'current_model_year': default_year,
        };
        active_context_map[mapside[1]] = active_context;

        active_context.map.style().set('cursor', 'crosshair');
        active_context.visParams = {
            min: 0.0,
            max: 100.0,
            palette: legend_styles[default_legend_style],
        };

        var panel = ui.Panel({
            layout: ui.Panel.Layout.flow('vertical'),
            style: {
                'position': "middle-"+mapside[1],
                'backgroundColor': 'rgba(255, 255, 255, 0.4)'
            }
        });

        var default_control_text = mapside[1]+' controls';
        var controls_label = ui.Label({
            value: default_control_text,
            style: {
                backgroundColor: 'rgba(0, 0, 0, 0)',
            }
        });
        panel.add(controls_label);

        var data_model_panel = ui.Panel({
            layout: ui.Panel.Layout.Flow('vertical'),
            style: {
                padding: '0px',
                backgroundColor: 'rgba(255, 255, 255, 0.4)'
            }
        });

        data_model_panel.add(ui.Label({
            value: 'Data and model year',
            style:{'backgroundColor': 'rgba(0, 0, 0, 0)'}
        }));


        var select_widget_list = [];
        var model_select_index = 1;
        var image_select_index = 0;

        var model_year = ui.Slider({
            min: 2019,
            max: 2021,
            value: default_year,
            step: 1,
            disabled: true,
            onChange: function (new_year) {

                if (active_context.last_layer !== null) {
                    active_context.map.remove(active_context.last_layer);
                }
                if (active_context.raster_type == 'model') {
                  var model_id = select_widget_list[
                    model_select_index].getValue();
                  var term_list = model_term_map[model_id];
                  var new_model = make_rangeland_model(
                    model_id, term_list, new_year);
                  active_context.raster = new_model;
                  active_context.last_layer = active_context.map.addLayer(
                      active_context.raster, active_context.visParams);
                  if (validation_check.getValue()) {
                    // retrigger graph drawing
                    validation_check.setValue(false);
                    validation_check.setValue(true);
                  }
                } else {
                  // update the image
                  var image_id = select_widget_list[
                    image_select_index].getValue();
                  build_images_by_year(new_year);
                  active_context.raster = global_image_dict[image_id].clip(
                    gobi_poly);
                  active_context.last_layer = active_context.map.addLayer(
                      active_context.raster, active_context.visParams);
                }
                active_context.current_model_year = new_year;

            }
          }
        );
        data_model_panel.add(model_year);

        active_context.model_year = model_year;

        var select_placeholder_list = ['Select model data ...', 'Select rangeland model ...'];
        [[global_image_dict, 'image'], [global_model_dict, 'model']].forEach(
            function (payload, index) {
                var local_image_dict = payload[0];
                var image_type = payload[1];
                var select = ui.Select({
                    placeholder: select_placeholder_list[index],
                    items: Object.keys(local_image_dict).sort(),
                    onChange: function(key, self) {
                        active_context.raster_type = image_type;
                        active_context.active_map_layer_id = key;
                        self.setDisabled(true);
                        active_context.validation_check.setDisabled(true);
                        active_context.model_year.setDisabled(true);
                        var original_value = self.getValue();
                        self.setPlaceholder('(loading ...) ' + original_value);
                        var other_index = (index+1)%2;
                        select_widget_list[other_index].setValue(null, false);
                        select_widget_list[other_index].setPlaceholder(
                            select_placeholder_list[other_index], false);
                        self.setValue(null, false);

                        active_context.map.centerObject(gobi_poly);
                        if (active_context.validation_layer !== null)  {
                            active_context.map.remove(
                                active_context.validation_layer);
                            active_context.validation_layer = null;
                            active_context.validation_check.setValue(
                                false, false);
                        }

                        if (active_context.last_layer !== null) {
                            active_context.map.remove(active_context.last_layer);
                            min_val.setDisabled(true);
                            max_val.setDisabled(true);
                        }
                        if (local_image_dict[key] === '') {
                            self.setValue(original_value, false);
                            self.setDisabled(false);
                            return;
                        }
                        active_context.raster = local_image_dict[key].clip(
                          gobi_poly);
                        var mean_reducer = ee.Reducer.percentile(
                            [10, 90], ['p10', 'p90']);
                        var meanDictionary = active_context.raster.reduceRegion({
                            reducer: mean_reducer,
                            geometry: active_context.map.getBounds(true),
                            scale: 300,
                            bestEffort: true,
                        });

                        ee.data.computeValue(meanDictionary, function (val) {
                            active_context.visParams = {
                                min: val.B0_p10,
                                max: val.B0_p90,
                                palette: active_context.visParams.palette,
                            };
                            active_context.last_layer = active_context.map.addLayer(
                                active_context.raster, active_context.visParams);

                            min_val.setValue(
                              active_context.visParams.min, false);
                            max_val.setValue(
                              active_context.visParams.max, false);
                            min_val.setDisabled(false);
                            max_val.setDisabled(false);
                            self.setValue(original_value, false);
                            self.setDisabled(false);

                            active_context.model_year.setDisabled(false);
                            if (image_type === 'model') {
                              active_context.validation_check.setDisabled(
                                false);
                            }
                        });
                        if (active_context.gobi_poly_loaded == false) {
                          active_context.map.addLayer(gobi_poly);
                          active_context.gobi_poly_loaded = true;
                        }
                      }
            });
            select_widget_list.push(select);
            data_model_panel.add(select);
        });

        var validation_check = ui.Checkbox({
            label: 'compare validation data',
            value: false,
            disabled: true,
            onChange: function (checked, self) {
              if (checked) {
                var validation_collection = active_context.raster.sampleRegions({
                    collection: validation_points,
                    geometries: true,
                    scale: 10,
                });
                var model_property_str = property_by_model[active_context.active_map_layer_id];

                if (active_context.chart_panel !== null) {
                  active_context.map.remove(active_context.chart_panel);
                }
                active_context.chart_panel = ui.Panel({
                  layout: ui.Panel.Layout.flow('vertical'),
                    style: {
                        'position': "top-"+mapside[1],
                        'backgroundColor': 'rgba(255, 255, 255, 0.4)'
                    }
                });

                var filtered_validation_collection = validation_collection.filter(ee.Filter.eq('Year', active_context.current_model_year));
                var chart =
                  ui.Chart.feature
                    .byFeature({
                      features: filtered_validation_collection,
                      xProperty: model_property_str,
                      yProperties: ['B0'],
                    })
                    .setChartType('ScatterChart')
                    .setOptions({
                      title: active_context.current_model_year + ' - Observed '+model_property_str+' vs ' + active_context.active_map_layer_id,
                      hAxis:
                          {title: 'Observed', titleTextStyle: {italic: false, bold: true}},
                      vAxis: {
                        title: 'Modelled Value',
                        titleTextStyle: {italic: false, bold: true}
                      },
                      pointSize: 3,
                      colors: ['009900'],
                    });
                active_context.map.add(active_context.chart_panel);
                active_context.chart_panel.add(chart);
                var agb_vs_b0_color = ee.Dictionary({
                    1: 'blue',
                    0: 'red',
                });
                var max_radius = 20;
                var visualized_validation_collection = filtered_validation_collection.map(function (feature) {
                    return feature.set('style', {
                        pointSize: ee.Number(feature.get(model_property_str)).subtract(feature.get('B0')).abs().divide(max_radius).min(max_radius),
                        color: agb_vs_b0_color.get(ee.Number(feature.get(model_property_str)).lt(ee.Number(feature.get('B0')))),
                    });
                });
                visualized_validation_collection = visualized_validation_collection.style({
                    styleProperty: 'style',
                    neighborhood: max_radius*2,
                });

                active_context.validation_layer = active_context.map.addLayer(
                    visualized_validation_collection);
              } else {
                active_context.map.remove(active_context.chart_panel);
                active_context.map.remove(active_context.validation_layer);
              }
            }
        });
        active_context.validation_check = validation_check;
        data_model_panel.add(validation_check);

        panel.add(data_model_panel);

        var min_val = ui.Textbox(0, 0, function (value) {
            active_context.visParams.min = +(value);
            updateVisParams();
        });
        min_val.setDisabled(true);

        var max_val = ui.Textbox(100, 100, function (value) {
            active_context.visParams.max = +(value);
            updateVisParams();
        });
        max_val.setDisabled(true);

        active_context.point_val = ui.Textbox('nothing clicked');
        function updateVisParams() {
            if (active_context.last_layer !== null) {
                active_context.last_layer.setVisParams(active_context.visParams);
            }
        }
        active_context.updateVisParams = updateVisParams;
        var range_button = ui.Button('Detect Range', function (self) {
            self.setDisabled(true);
            var base_label = self.getLabel();
            self.setLabel('Detecting...');
            var mean_reducer = ee.Reducer.percentile([10, 90], ['p10', 'p90']);
            var meanDictionary = active_context.raster.reduceRegion({
              reducer: mean_reducer,
              geometry: active_context.map.getBounds(true),
              bestEffort: true,
            });
            ee.data.computeValue(meanDictionary, function (val) {
              min_val.setValue(val.B0_p10, false);
              max_val.setValue(val.B0_p90, true);
              self.setLabel(base_label);
              self.setDisabled(false);
            });
        });

        panel.add(ui.Label({
            value: 'min',
            style:{'backgroundColor': 'rgba(0, 0, 0, 0)'}
        }));
        panel.add(min_val);
        panel.add(ui.Label({
            value: 'max',
            style:{'backgroundColor': 'rgba(0, 0, 0, 0)'}
        }));
        panel.add(max_val);
        panel.add(range_button);
        panel.add(ui.Label({
          value: 'picked point',
          style: {'backgroundColor': 'rgba(0, 0, 0, 0)'}
        }));
        panel.add(active_context.point_val);
        panel_list.push([panel, min_val, max_val, active_context]);
        active_context.map.add(panel);

        function build_legend_panel() {
            var makeRow = function(color, name) {
                var colorBox = ui.Label({
                    style: {
                        backgroundColor: '#' + color,
                        padding: '4px 25px 4px 25px',
                        margin: '0 0 0px 0',
                        position: 'bottom-center',
                    }
                });
                var description = ui.Label({
                    value: name,
                    style: {
                        margin: '0 0 0px 0px',
                        position: 'top-center',
                        fontSize: '10px',
                        padding: 0,
                        border: 0,
                        textAlign: 'center',
                        backgroundColor: 'rgba(0, 0, 0, 0)',
                    }
                });

                return ui.Panel({
                    widgets: [colorBox, description],
                    layout: ui.Panel.Layout.Flow('vertical'),
                    style: {
                        backgroundColor: 'rgba(0, 0, 0, 0)',
                    }
                });
            };

            var names = ['Low', '', '', '', 'High'];
            if (active_context.legend_panel !== null) {
                active_context.legend_panel.clear();
            } else {
                active_context.legend_panel = ui.Panel({
                    layout: ui.Panel.Layout.Flow('horizontal'),
                    style: {
                        position: 'top-center',
                        padding: '0px',
                        backgroundColor: 'rgba(255, 255, 255, 0.4)'
                    }
                });
                active_context.legend_select = ui.Select({
                    items: Object.keys(legend_styles),
                    placeholder: default_legend_style,
                    onChange: function(key, self) {
                        changeColorScheme(key, active_context);
                    }
                });
                active_context.map.add(active_context.legend_panel);
            }
            active_context.legend_panel.add(active_context.legend_select);
            for (var i = 0; i<5; i++) {
                var row = makeRow(active_context.visParams.palette[i], names[i]);
                active_context.legend_panel.add(row);
            }
        }

        active_context.map.setControlVisibility(false);
        active_context.map.setControlVisibility({"mapTypeControl": true});
        build_legend_panel();
        active_context.build_legend_panel = build_legend_panel;
    }); // end map definition

    var clone_to_right = ui.Button(
      'Use this range in both windows', function () {
          var active_context = active_context_map.right;
          active_context.visParams.min = panel_list[0][1].getValue();
          active_context.visParams.max = panel_list[0][2].getValue();
          panel_list[1][1].setValue(active_context.visParams.min, false);
          panel_list[1][2].setValue(active_context.visParams.max, false);
          active_context.updateVisParams();

    });
    var clone_to_left = ui.Button(
      'Use this range in both windows', function () {
          var active_context = active_context_map.left;
          active_context.visParams.min = panel_list[1][1].getValue();
          active_context.visParams.max = panel_list[1][2].getValue();
          panel_list[0][1].setValue(active_context.visParams.min, false);
          panel_list[0][2].setValue(active_context.visParams.max, false);
          active_context.updateVisParams();
    });

    panel_list.forEach(function (panel_array) {
      var map = panel_array[3].map;
      map.onClick(function (obj) {
        var point = ee.Geometry.Point([obj.lon, obj.lat]);
        [panel_list[0][3], panel_list[1][3]].forEach(function (active_context) {
          if (active_context.last_layer !== null) {
            active_context.point_val.setValue('sampling...');
            var point_sample = active_context.raster.sampleRegions({
              collection: point,
              scale: 10,
            });
            ee.data.computeValue(point_sample, function (val) {
              if (val.features.length > 0) {
                active_context.point_val.setValue(val.features[0].properties.B0.toString());
                if (active_context.last_point_layer !== null) {
                  active_context.map.remove(active_context.last_point_layer);
                }
                active_context.last_point_layer = active_context.map.addLayer(
                  point, {'color': '#FF00FF'});
              } else {
                active_context.point_val.setValue('nodata');
              }
            });
          }
        });
      });
    });

    panel_list[0][0].add(clone_to_right);
    panel_list[1][0].add(clone_to_left);
} // end ui definition

/*
Object.keys(veg_cover_model).forEach(function (model_id) {
    var table = veg_cover_model[model_id];
    var table_to_list = table.toList(table.size());
    //Create Carbon Regression Image based on table coefficients
    table_to_list.evaluate(function (term_list) {
        model_term_map[model_id] = term_list;
        [2019, 2020, 2021].forEach(function (model_year) {
          var model_image = make_rangeland_model(model_id, term_list, model_year);
          var model_filename = model_id + '_' + model_year;

          Export.image.toCloudStorage({
            'image': model_image,
            'description': model_filename,
            'bucket': 'ecoshard-root',
            'fileNamePrefix': 'rangeland_model/'+model_filename,
            'maxPixels': 1e12,
            'fileFormat': 'GeoTIFF',
            'region': gobi_poly,
            'crs': 'epsg:4326',
            'scale': 10,
        });
      });
    });
});
*/
