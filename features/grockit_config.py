config = {
	"entities" : {
		"Outcomes" : {
			"feature_metadata" : {
				"correct" : {
					"categorical" : True,
					"numeric" : False,
					"categorical_filter": True,
				}, 
			},
			"one-to-one" : [],
			"included_row_functions":[],
			"excluded_row_functions":[]
		},

		"Users" : {
			"feature_metadata" : {
				
			},
			"one-to-one" : []
		},

		"GameTypes" : {
			"feature_metadata" : {
				"game_type" : {
					"categorical" : True,
					"numeric" : False,
					"categorical_filter": True,
				}, 
			}
		},

		"Tracks" : {
			"track" : {
				"categorical" : True,
				"numeric" : False,
				"categorical_filter": True,
			}, 
		},

		"Subtracks" : {
			"subtrack" : {
				"categorical" : True,
				"numeric" : False,
			}, 
		},

		"Questions" : {
			"feature_metadata" : {
				"question_type" : {
					"categorical" : True,
					"numeric" : False,
					"categorical_filter": True,
				}, 
			}
		},

		"Tags" : {
			"feature_metadata" : {
				"tag_name" : {
					"categorical" : True,
					"numeric" : False
				}
			}
		},

		"Groups" : {
			"feature_metadata" : {
				"group" : {
					"categorical" : True,
					"categorical_filter" : True,
					"numeric" : False
				}
			}
		},
	},

	"max_categorical_filter" : 2,
}