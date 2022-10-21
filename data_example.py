'''
A data format example for "matched_traj.pkl"

The file "matched_traj.pkl" which is readed in "calculate_speed.py" is the map-matched historical trajectories.
It also involves original GPS points thus is not open access.
If you' re interested in running "calculate_speed.py", you have to prepare your map-matched trajectories.
Here shows the data format of each item in "matched_traj.pkl" for your convenience.
'''

lon, lat = 111, 22

{
 'index': 64074,
 'path': [
          [901,   # 路网中的edge_id, 匹配路径的第一条路
           [      # 匹配到这条路上的所有轨迹点
            {'order': 0,  # 轨迹点在原始经纬度轨迹中的索引
             'orig_point': [lon, lat, 6467.0],  # 匹配后投影到路上的轨迹点[lon, lat, t]
             'point': [lon, lat, 6467.0]},      # 原始轨迹点
            {'order': 1,
             'orig_point': [lon, lat, 6478.0],
             'point': [lon, lat, 6478.0]},
            {'order': 2,
             'orig_point': [lon, lat, 6547.0],
             'point': [lon, lat, 6547.0]}
           ]      
          ],
          [325,   # 路网中的edge_id, 匹配路径的第二条路
           [
            {'order': 3,
             'orig_point': [lon, lat, 6557.0],
             'point': [lon, lat, 6557.0]},
            {'order': 4,
             'orig_point': [lon, lat, 6567.0],
             'point': [lon, lat, 6567.0]},
            {'order': 5,
             'orig_point': [lon, lat, 6578.0],
             'point': [lon, lat, 6578.0]}
           ]
          ]
        ],
 'start_end_portion': (
    0.1663753030146705,  # 第一条路被通过的比例(即第一条路上, 第一个轨迹点及之后的部分的占比)
    0.5463860908881429   # 最后一条路被通过的比例(即最后一条路上, 最后一个轨迹点及之前的部分的占比)
  )  
}