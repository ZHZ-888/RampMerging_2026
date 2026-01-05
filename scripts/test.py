dic_platoon = {'mav38': ['mav38',
  'mhv48',
  'mhv65',
  'mhv83',
  'mhv93',
  'mhv103',
  'mhv121',
  'mhv131',
  'mhv185',
  'mhv198',
  'mhv268'],
 'mav278': ['mav278',
  'mhv288',
  'mhv298',
  'mhv404',
  'mhv443',
  'mhv482',
  'mhv492',
  'mhv642',
  'mhv653',
  'mhv663']}

veh_to_leader = {
    veh_id: leader_id
    for leader_id, members in dic_platoon.items()
    for veh_id in members
}

print(veh_to_leader)