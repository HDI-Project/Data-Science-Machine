config = {
	"entities" : {
		"Outcomes" : {
			"feature_metadata" : {
				"label" : {
					"categorical" : True,
					"numeric" : False,
					"categorical_filter" : True
				}
			},
			"one-to-one" : []
		},

		"Users" : {
			"feature_metadata" : {
				"age_range" : {
					"categorical" : True,
					"numeric" : False,
					"categorical_filter" : True
				}
			},
			"one-to-one" : []
		},

		"Behaviors" : {
			"excluded_row_functions" : ["weekday", "month"]
		},

		"Actions" : {
			"feature_metadata" : {
				"name" : {
					"categorical" : True,
					"numeric" : False,
					"categorical_filter" : True
				}
			},
			"excluded_row_functions" : ["length"]
		}

	}
}