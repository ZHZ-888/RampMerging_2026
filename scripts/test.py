#%%
ls_leader = ['mav3287', 'mav3038', 'mav2744']

ls_ihA = ['mhv3305', 'mav3287', 'mhv3171', 'mav3038', 'mav2744']
ls_ihA_asc = ls_ihA[::-1]
print(ls_ihA_asc)

#%%
for leader in ls_leader:
    idx = ls_ihA_asc.index(leader)
    print(idx)