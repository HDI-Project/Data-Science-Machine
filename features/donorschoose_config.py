config = {
	"entities" : {
		"Projects" : {
			"feature_metadata" : {

			},
			"one_to_one" : {
				"Outcomes" : True,
				"Essays" : True,
			},

			"excluded_predict_entities" : ["Donations", "Outcomes"],

			'excluded_agg_entities' : ["Donations"],
		},

		"Donors" : {
			'excluded_agg_entities' : ["Donations"]
		},

		"Schools" : {
			"feature_metadata" : {
				"school_city" : {
					"categorical" : True,
				},
				"school_state" : {
					"categorical" : True,
				},
				"poverty_level" : {
					"categorical" : True,
				},
				"city" : {
					"categorical" : True,
				},
			},

			"make_intervals": {
				"Projects" : {
					"delta_days" : 30,
					"n_intervals" : 12,
				}
			}
		},

		"Teachers" : {
			"make_intervals": {
				"Projects" : {
					"delta_days" : 30,
					"n_intervals" : 12,
				}
			}
		},

		"Donations" : {
			"excluded_row_functions" : ["weekday", "month"]
		}
	},
}