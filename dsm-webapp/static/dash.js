$(function () {    
    $(".month-select .btn").click(function(){
        var month_data = $(this).data("month").split("/")
        var month = month_data[0]
        var year = month_data[1]
        $('#monthly-data').fadeOut(0)
        $('#monthly-graph').fadeOut(0)
        $("#reset").fadeIn(0)
        show_month(month,year)
    })

    $("#reset").click(function(){
        $("#reset").fadeOut(0)
        $("#daily-graph").fadeOut(0)
        $('#monthly-data').fadeIn(0)
        show_all_months()
    })

    $("#reset").click()
});

function show_month(month, year){
    $.get("/daily", {"token": $("#header").data("client-token"), "month":month, "year": year}, function(res){
        var data = $.map(res.data, function(v, i){
            return [[Date.UTC(v.year, v.month-1, v.day), v.profit/100]] //date.utc month is zero indexed
        })

        $("#total_orders").text(res.metrics.total_orders)
        $("#profit").text("$"+(res.metrics.profit/100).formatMoney(2))
        $("#gross_earnings").text("$"+(res.metrics.gross_earnings/100).formatMoney(2))
        $("#amazon_total").text("$"+(res.metrics.amazon_total/100).formatMoney(2))

        $('#daily-graph').fadeIn(0).highcharts({
            chart: {
                type: 'column'
            },
            title: {
                text: 'Profits for ' + month + "/" + year
            },
            xAxis: {
                type: 'datetime',
                dateTimeLabelFormats: {
                    day: '%b %e'
                },
                title: {
                    text: 'Date'
                }
            },
            yAxis: {
                title: {
                    text: 'Profit'
                },
            },
            tooltip: {
                headerFormat: '<b>{series.name}</b><br>',
                pointFormat: '{point.x: %b %d, %Y}: ${point.y:.2f}'
            },

            legend : {
                enabled:false
            },
            series: [{
                name: 'Profit',
                // Define the data points. All series have a dummy year
                // of 1970/71 in order to be compared on the same x axis. Note
                // that in JavaScript, months start at 0 for January, 1 for February etc.
                data: data
            }]
        });
    })
}

function show_all_months(){
    $.get("/monthly", {"token": $("#header").data("client-token")}, function(res){
        var data = $.map(res.data, function(v, i){
            return [[Date.UTC(v.year, v.month-1), v.profit/100]] //date.utc month is zero indexed
        })

        $("#total_orders").text(res.metrics.total_orders)
        $("#profit").text("$"+(res.metrics.profit/100).formatMoney(2))
        $("#gross_earnings").text("$"+(res.metrics.gross_earnings/100).formatMoney(2))
        $("#amazon_total").text("$"+(res.metrics.amazon_total/100).formatMoney(2))

        $('#monthly-graph').fadeIn(0).highcharts({
            chart: {
                type: 'column'
            },
            title: {
                text: 'Profits by month'
            },
            xAxis: {
                type: 'datetime',
                dateTimeLabelFormats: { // don't display the dummy year
                    month: '%b \'%y',
                },
                title: {
                    text: 'Date'
                }
            },
            yAxis: {
                title: {
                    text: 'Profit'
                },
                min: 0
            },
            tooltip: {
                headerFormat: '<b>{series.name}</b><br>',
                pointFormat: '{point.x:%b \'%y}: ${point.y:.2f}'
            },

            legend : {
                enabled:false
            },
            series: [{
                name: 'Profits',
                // Define the data points. All series have a dummy year
                // of 1970/71 in order to be compared on the same x axis. Note
                // that in JavaScript, months start at 0 for January, 1 for February etc.
                data: data
            }]
        });
        
    })
}

//http://stackoverflow.com/questions/149055/how-can-i-format-numbers-as-money-in-javascript
Number.prototype.formatMoney = function(c, d, t){
var n = this, 
    c = isNaN(c = Math.abs(c)) ? 2 : c, 
    d = d == undefined ? "." : d, 
    t = t == undefined ? "," : t, 
    s = n < 0 ? "-" : "", 
    i = parseInt(n = Math.abs(+n || 0).toFixed(c)) + "", 
    j = (j = i.length) > 3 ? j % 3 : 0;
   return s + (j ? i.substr(0, j) + t : "") + i.substr(j).replace(/(\d{3})(?=\d)/g, "$1" + t) + (c ? d + Math.abs(n - i).toFixed(c).slice(2) : "");
 };