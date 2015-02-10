def make_features(table):
	
	for fp in feature_paths:
		for node in fp['path']:
			join_data = make_features(node)
			complete_data = complete_data.join(join_data)



	return complete_data
			