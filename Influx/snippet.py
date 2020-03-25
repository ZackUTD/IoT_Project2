  import torch
  from influxdb import DataFrameClient as df_client
  import numpy as np
  import pandas as pd
  
  client = df_client(host='10.176.67.83', port=8086)  # client that reads/writes db with pandas dataframes
  client.switch_database('team_3_test_offline')
  
  X_torch = None
	Y_torch = None

	query = 'select * from bearing where mw = ' + str(batch_no)

	results = client.query(query)

	if results:
		results_df = results['bearing']
		x_np = results_df.take([0,2,5], axis=1).to_numpy()  # currently only looking at gs, sr, and load  --> to numpy
		X_torch = torch.tensor(x_np).type(torch.FloatTensor).float()  --> to pytorch tensor
	
		y_np = results_df.take([1], axis=1).to_numpy().astype(float)
		Y_torch = torch.from_numpy(y_np).float()
