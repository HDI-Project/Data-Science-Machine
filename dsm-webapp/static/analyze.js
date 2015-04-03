$(function () { 
	// $('#intro-modal').modal()
	function set_model(score, using){
		// var model_names = ["SVM", "Random Forest", "Linear Regression"]
		$('.model-score').removeClass("label-success")
							.removeClass("label-warning")
							.removeClass("label-danger")

		var score = score*100
		if (score >= 90){
			$('.model-score').addClass("label-success")
		} else if (score >= 80){
			$('.model-score').addClass("label-warning")
		} else{
			$('.model-score').addClass("label-danger")
		}
		$('.prev-model-score').text($('.model-score').text())
		$('.model-score').text(score.toFixed(2))
		// $('.best-model').text(model_names.sort(function() {return 0.5 - Math.random()})[0])

		var feature_importance = using.sort(function(a,b) {return Math.abs(b.weight) - Math.abs(a.weight)})
		var select_vals = []
		console.log(score, feature_importance)
		$("#model-best-features").empty()
		$.each(feature_importance, function(i,v){
			var $li = $("<li data-weight='" + v.id + "'>" + v.name + ":  " +  v.weight.toFixed(2) + "</li>")
			$("#model-best-features").append($li)
			select_vals.push(v.id)
		})

		$('#feature-selector').val(select_vals).trigger("change");
	}

	function make_model(target, using){
		$('#model-modal').modal()
		payload = {
			"entity" : $("#entity-header").data("entity-name"),
			"target" : target,
			"using"  : using,
		}
		$.get("/model", payload, function(res){
			set_model(res.score, res.using)
			$('#model-modal').modal("hide")
			$("#refresh-model").addClass("hidden")
			$('#model-score').removeClass("hidden")
			$('#prev-model-score').removeClass("hidden")
			$('#model-summary-text').removeClass("hidden")
		})
	}

	$('.feature-item').click(function(){
		$('.feature-item').removeClass("selected")
		$(this).addClass("selected")

		$('#feature-eng-panel').removeClass("hidden")
		$('#feature-selector').select2();

		var feature_id = $(this).data("id")
		var feature_name = $(this).data("name")

		make_model(feature_id, [])


		$('.selected-feature').text(feature_name).data("id", feature_id)
	})


	$('#feature-selector').on("change", function (e) { 
		// $('#refresh-model').prop("disabled", false)
		$('#model-score').addClass("hidden")
		$('#prev-model-score').addClass("hidden")
		$('#refresh-model').removeClass("hidden")
		$('#model-summary-text').addClass("hidden")
		console.log("change");
	});



	$('#refresh-model').click(function(){
		var select_vals = $("#feature-selector").val()
		var target = $('.selected-feature').data("id")
		make_model(target, select_vals)
	})

	$('#show-more-features').click(function(){
		$(".feature-item.extra").toggleClass("hidden")
	})

	$('#toggle-advanced').click(function(){
		$("#advanced").toggleClass("hidden")
	})
})
