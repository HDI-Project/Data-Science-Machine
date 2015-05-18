config = {
	"entities" : {
		"Outcomes" : {
			"feature_metadata" : {
				"dropped_out" : {
					"categorical" : True,
					"numeric" : False,
					# "categorical_filter": True,
				}, 
			},
			"one-to-one" : ["Enrollment"],
			"included_row_functions":[],
			"excluded_row_functions":[]
		},

		"Users" : {
			"feature_metadata" : {
			},
			"one-to-one" : [],
		},

		"Courses" : {
			"feature_metadata" : {
			},
		},

		"Enrollments" : {
			"feature_metadata" : {

				"test" : {
					"ignore": True
				}
			},
			'excluded_predict_entities' : ["Outcomes"]
			# "train_filter" : [["test", "=", 0]],
		},

		"Log" : {
			"feature_metadata" : {
				# "source" : {
				# 	"categorical" : True,
				# 	"numeric" : False,
				# 	"categorical_filter" : True
				# }, 
				"test" : {
					"ignore" : True
				}
			},
		    "included_row_functions":[],
			# "train_filter" : [["test", "=", 0]],
		},

		"EventTypes" : {
			"feature_metadata" : {
				"event" : {
					"categorical" : True,
					"numeric" : False,
					"categorical_filter": True,
				}, 
			},
			'excluded_agg_entities' : ["Log"]
		},

		"Objects" : {
			"feature_metadata" : {
				"category" : {
					"categorical" : True,
					"numeric" : False,
					"categorical_filter" : True,
				}
			},
		},
	},

	"max_categorical_filter" : 2,
}