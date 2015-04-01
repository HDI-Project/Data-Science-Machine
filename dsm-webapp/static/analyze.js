$(function () { 
	$('#intro-modal').modal()
	function random_accuracy(){
		var model_names = ["SVM", "Random Forest", "Linear Regression"]
		$('.model-accuracy').removeClass("label-success")
							.removeClass("label-warning")
							.removeClass("label-danger")

		var accuracy = Math.floor(70+Math.random()*30)
		if (accuracy >= 90){
			$('.model-accuracy').addClass("label-success")
		} else if (accuracy >= 80){
			$('.model-accuracy').addClass("label-warning")
		} else{
			$('.model-accuracy').addClass("label-danger")
		}
		$('.model-accuracy').text(accuracy)
		$('.best-model').text(model_names.sort(function() {return 0.5 - Math.random()})[0])

		var feature_importance = $('#feature-selector').val().sort(function() {return 0.5 - Math.random()})
		$("#model-best-features").empty()
		$.each(feature_importance, function(i,v){
			$("#model-best-features").append("<li>" + v + "</li>")
		})

	}

	function make_model(target, using){
		payload = {
			"entity" : $("#entity-header").data("entity-name"),
			"target" : target,
			"using"  : using,
		}
		$.get("/model", payload, function(res){
			set_model(target, using)
		})
	}

	var input_features = {}

	$('.feature-item').click(function(){
		$('.feature-item').removeClass("selected")
		$(this).addClass("selected")

		$('#feature-eng-panel').removeClass("hidden")
		$('#feature-selector').select2();

		var feature_id = $(this).data("id")

		make_model(feature_id, [])

		console.log(feature_id)

		// $('.selected-feature').text(feature_name)

		// if (!input_features[feature_name]){
		// 	input_features[feature_name] = features.sort(function() {return 0.5 - Math.random()}).slice(0, Math.floor(2+5*Math.random()))
		// }

		// $('#feature-selector').val(input_features[feature_name]).trigger("change"); 
		// $("#refresh-model").click()		
	})

	$("#refresh-model").click(function(){
		var progress = 0;
		function progress_loop () {          
		   setTimeout(function () {   
		      $('#model-progress').width(progress+"%")
		      if (progress >= 100){
				random_accuracy()
				// $(this).prop("disabled", "disabled")
				$("#refresh-model").addClass("hidden")
				$('#model-accuracy').removeClass("hidden")
				$('#model-summary-text').removeClass("hidden")
				$('#model-modal').modal("hide")
				$('#model-progress').width(0+"%")
		      } 
		   	  else {
		      	progress_loop();
				progress += Math.random()*30*Math.random()
				progress = Math.min(100, progress)
		      }

		      if (progress >= 66){
		      	$('#model-status').text("Testing SVM...")
		   	  } else if (progress >= 33){
		   	  	$('#model-status').text("Testing Linear Regression...")
		   	  } else {
		   	  	$('#model-status').text("Testing Random Forest...")
		   	  }

		   }, 150)
		}

		$('#model-modal').modal()
		progress_loop()

	})

	$('#feature-selector').on("change", function (e) { 
		// $('#refresh-model').prop("disabled", false)
		$('#model-accuracy').addClass("hidden")
		$('#refresh-model').removeClass("hidden")
		$('#model-summary-text').addClass("hidden")
		console.log("change");
	});

	$('#show-more-features').click(function(){
		$(".feature-item.extra").toggleClass("hidden")
	})

	$('#toggle-advanced').click(function(){
		$("#advanced").toggleClass("hidden")
	})
})
