basic_link = '../DataSet/delicious'
OrgSet_link = basic_link + '/OrgSet'
MapSet_link = basic_link + '/MapSet'
MainSet_link = basic_link + '/MainSet'

"""------------------------------ OrgSet -------------------------------"""
logs_filename = OrgSet_link + '/user_taggedartists-timestamps.dat'
# social_filename = OrgSet_link + '/user_friends.dat'
tags_filename = OrgSet_link + '/tags.dat'
social_filename = OrgSet_link + '/user_contacts.dat'

"""------------------------------ MapSet -------------------------------"""
map_link = MapSet_link + '/map_%s'

"""------------------------------ MainDataSet_link -------------------------------"""
logs_link = MainSet_link + '/logs'
sub_logs_link = MainSet_link + '/sub_logs'
user_context_link = MainSet_link + '/user_context'
item_context_link = MainSet_link + '/item_context'
user_selected_link = MainSet_link + '/user_selected'
bandit_data_link = MainSet_link + '/bandit_data_%d' + '.json'

user_clusterID_link = MainSet_link + '/user_clusterID'
social_mat_link = MainSet_link + '/social_mat'
result_link = MainSet_link + '/result_factorUCB'

