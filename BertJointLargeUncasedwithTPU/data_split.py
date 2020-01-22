import pandas as pd

def extract_target_variable(df, short = True):
    if short:
        short_answer = []
        for i in range(len(df)):
            short = df['annotations'][i][0]['short_answers']
            if short == []:
                yes_no = df['annotations'][i][0]['yes_no_answer']
                if yes_no == 'NO' or yes_no == 'YES':
                    short_answer.append(1)
                else:
                    short_answer.append(0)
            else:
                short = short[0]
                st = short['start_token']
                et = short['end_token']
                s_len = et - st
                #short_answer.append(f'{st}'+':'+f'{et}')
                short_answer.append(s_len)
        short_answer = pd.DataFrame({'short_answer': short_answer})
        return short_answer
    else:
        long_answer = []
        for i in range(len(df)):
            long = df['annotations'][i][0]['long_answer']
            if long['start_token'] == -1:
                long_answer.append(0)
            else:
                st = long['start_token']
                et = long['end_token']
                l_len = et - st
                long_answer.append(l_len)
        long_answer = pd.DataFrame({'long_answer': long_answer})
        return long_answer
    
def make_percentile(pixel_count, q33, q66):    
    if pixel_count == 0.0:
        percentile = 0
    elif pixel_count > 0.0 and pixel_count <= q33:
        percentile = 1
    elif pixel_count > q33 and pixel_count <= q66:
        percentile = 2
    elif pixel_count > q66:
        percentile = 3
    return percentile