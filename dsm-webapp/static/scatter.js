$(function(){
	$('#feature-x').select2();
	$('#feature-y').select2();

	$("#go-features").click(function(){
		console.log()
		console.log($('#feature-y').val())

		payload = {
			"entity" : $("#entity-header").data("entity-name"),
			"x" : $('#feature-x').val(),
			"y" : $('#feature-y').val(),
		}

		$("#loader-container").removeClass("hidden")
		$("#graph").addClass("hidden")

		$.get("/scatter-data", payload, function(res){
			console.log(res)
			$("#loader-container").addClass("hidden")
			$("#graph").removeClass("hidden")
			var height= $("#graph-container").height();
			var width= $("#graph-container").width();

			var vals = $.map(res.data, function(p,i){ return p[0] })
			var max_val = Math.max.apply(null,vals);
			var min_val = Math.min.apply(null,vals)

			var data = $.map(res.data, function(p,i){
				var norm_val = (p[0]-min_val)/max_val

				console.log(norm_val)

				return {
					"x":p[0],
					"y":p[1],
					// fillColor: '#444',
					// color: '#444',
					// marker:{ fillColor: '#444', color:}
				}
			})

			console.log(data)

			$('#graph').highcharts({
			        chart: {
			            type: 'scatter',
			            zoomType: 'xy',
			            height:height,
			            width:width
			        },
			        title: {
			            text: res.title
			        },
			        xAxis: {
			            title: {
			                enabled: true,
			                text: res.x_axis
			            },
			            startOnTick: true,
			            endOnTick: true,
			            showLastLabel: true
			        },
			        yAxis: {
			            title: {
			                text: res.y_axis
			            }
			        },
			        plotOptions: {
			            scatter: {
			                marker: {
			                    radius: 5,
			                    states: {
			                        hover: {
			                            enabled: true,
			                            lineColor: 'rgb(100,100,100)'
			                        }
			                    }
			                },
			                states: {
			                    hover: {
			                        marker: {
			                            enabled: false
			                        }
			                    }
			                },
			                tooltip: {
			                    headerFormat: '<b>{series.name}</b><br>',
			                    pointFormat: res.x_axis + ': <b>{point.x}</b> <br>' + res.y_axis + ': <b>{point.y}</b>'
			                },
			                // marker: {
			                //     fillColor: '#FFFFFF',
			                //     lineWidth: 2,
			                //     lineColor: null // inherit from series
			                // }
			            }
			        },
			        series: [{
			            name: 'Data',
			            color: 'rgba(223, 83, 83, .5)',
			            data: data.slice(0,999)
			        }]
			    });
		})
	})
})