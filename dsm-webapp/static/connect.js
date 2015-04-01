$(function () { 
	$('#connect-submit').click(function(){
		$('#connect-form').hide()
		$('#connect-progress-container').removeClass("hidden")
		
		var progress = 0;
		function progress_loop () {          
		   setTimeout(function () {   
		      $('#connect-progress').width(progress+"%")
		      if (progress >= 100){
		      	window.location = "/entity"
		      } else {
		      	progress_loop();
				progress += Math.random()*15
				progress = Math.min(100, progress)
		      }
		   }, 200)
		}

		progress_loop()
	})

})