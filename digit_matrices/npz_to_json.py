# write npz to json
import json
import numpy as np

all_prob = np.load(f'all_problems_symb.npz', allow_pickle=True)

all_problems = all_prob['all_problems'].item()
all_prob_types = all_problems.keys()

all_data = {}
for prob_type in all_prob_types:
    prob_type_data = []
    for p, prob in enumerate(all_problems[prob_type]['prob']):
        # convert problem to string
        prompt = ''
        for r in range(3):
            for c in range(3):
                prompt+='['
                if r==2 and c==2:
                    for i in range(len(prob[1][c])):
                        prompt += '&nbsp&nbsp'
                    if i < (len(prob[1][c]) - 1):
                        prompt += ' '
                else:
                    for i, entry in enumerate(prob[r][c]):
                        if entry == -1:
                            prompt += '&nbsp&nbsp'
                        else:
                            prompt += str(entry)

                        if i < (len(prob[r][c]) - 1):
                            prompt += ' '
                prompt+=']'
                if c < 2:
                    prompt += ' &nbsp '
                if r < 2 and c == 2:
                    prompt += '<br>'
        # print(prompt)
        choices = all_problems[prob_type]['answer_choices'][p]
    # for choices in all_problems[prob_type]['answer_choices']:
        options = []
        for a in range(8):
            option = '['
            for i, c in enumerate(choices[a]):
                option += str(c)
                if i < (len(choices[a]) - 1):
                    option += ' '
            if len(choices[a]) == 0:
                option += '&nbsp&nbsp'
            option += ']'
            options.append(option)
    
        prob_type_data.append({'prompt':prompt,\
                                'options':options,\
                                'correct':int(all_problems[prob_type]['correct_ind'][p]),\
                                'prob_ind':p})

    all_data[prob_type] = prob_type_data

print(all_data)

json_string=json.dumps(all_data, indent=2)

js_fname = 'all_problems_symb.js'
with open(js_fname, 'w') as f:
    f.write('var all_problems = ' + json_string)