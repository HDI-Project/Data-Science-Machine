$(function(){
	$("#go-cluster").click(function(){
		payload = {
			"entity" : $("#entity-header").data("entity-name"),
			"k" : $('#k-clusters').val()
		}

		$("#loader-container").removeClass("hidden")
		$("#graph").addClass("hidden")

		$.get("/cluster-data", payload, function(res){
			console.log(res)
			$("#loader-container").addClass("hidden")
			$("#graph").removeClass("hidden")
			var height= $("#graph-container").height();
			var width= $("#graph-container").width();

			$('#k-clusters').val(res.k)

			var data = $.map(res.clusters, function(p,i){

				return {
					"name":i,
					"data":p,
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
					   lineWidth: 0,
					   minorGridLineWidth: 0,
					   lineColor: 'transparent',
					   labels: {
					       enabled: false
					   },
					   minorTickLength: 0,
					   tickLength: 0
			        },
			        yAxis: {
					    labels: {
					       enabled: false
					    },
			        	gridLineWidth: 0,
  					   	minorGridLineWidth: 0
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
			                    headerFormat: 'Cluster <b>{series.name}</b><br>',
			                    pointFormat: ""
			                },
			                // marker: {
			                //     fillColor: '#FFFFFF',
			                //     lineWidth: 2,
			                //     lineColor: null // inherit from series
			                // }
			            }
			        },
			        series: data
			    });
		})
	})
})