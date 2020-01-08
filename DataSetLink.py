basic_link = '../DataSet/delicious'
OrgSet_link = basic_link + '/OrgSet'
MapSet_link = basic_link + '/MapSet'
MainSet_link = basic_link + '/MainSet'
ResSet_link = basic_link + '/ResSet'
"""------------------------------ OrgSet -------------------------------"""
logs_filename = OrgSet_link + '/raw_data.dat'
social_filename = OrgSet_link + '/user_contacts.dat'
tags_filename = OrgSet_link + '/tags.dat'

"""------------------------------ MapSet -------------------------------"""
map_link = MapSet_link + '/map_%s'
user_clusterID_link = MainSet_link + '/user_clusterID'
social_mat_link = MainSet_link + '/social_mat_%d'

"""------------------------------ MainSet -------------------------------"""
logs_link = MainSet_link + '/logs'
sub_logs_link = MainSet_link + '/sub_logs'
user_context_link = MainSet_link + '/user_context'
item_context_link = MainSet_link + '/item_context'
user_selected_link = MainSet_link + '/user_selected'
bandit_data_link = MainSet_link + '/bandit_data_%d' + '.json'

"""------------------------------ ResSet -------------------------------"""
result_link = ResSet_link + '/result_%s'

