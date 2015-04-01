$(function () { 
	$('.entity-select').click(function(){
		$('.entity-select').removeClass("selected")
		$(this).addClass("selected")
		$('#entity-submit').removeClass("hidden")
	})

	$('#entity-submit').click(function(){
		$('#entity-form').hide()
		$('#entity-progress-container').removeClass("hidden")
		var entity_name = $('.selected .entity-name').text()
		var entity_count = $('.selected .entity-count').text()

		$('#analyze-name').text(entity_name)
		$('#analyze-count').text(entity_count)

		var progress = 0;
		function progress_loop () {          
		   setTimeout(function () {   
		      $('#entity-progress').width(progress+"%")
		      if (progress >= 100){
		      	window.location = "/analyze?entity=" + entity_name
		      } else {
		      	progress_loop();
				progress += Math.random()*(10*Math.random())
				progress = Math.min(100, progress)
		      }
		   }, 10)
		}

		progress_loop()
	})
})