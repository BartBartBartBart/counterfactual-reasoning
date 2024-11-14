import json

all_permuted = {}
# Opening JSON file 
for i in [1,2,5,10,20,'symb']:
    with open(f'problems/nogen/all_prob_{i}_7_human.json', 'r') as f:
        # returns JSON object as  
        # a dictionary 
        data = json.load(f)
        all_permuted[f'np_{i}'] = data
   
print(all_permuted.keys())

all_prob_json_string = json.dumps(all_permuted, indent=2)

# Write to js script
js_fname = f'./problems/nogen/all_prob_all_permuted_7_human.js'
js_fid = open(js_fname, 'w')
js_fid.write('var all_problems = ' + all_prob_json_string)
js_fid.close()

json_fname = f'./problems/nogen/all_prob_all_permuted_7_human.json'
json_fid = open(json_fname, 'w')
json_fid.write(all_prob_json_string)
json_fid.close()