from pylsl import StreamInfo, StreamOutlet
from pylsl.pylsl import StreamInlet, resolve_byprop


## -----------------------------------------------------------------
## 发送数据 ---------------------------------------------------------

info = StreamInfo(
    name='meta_feedback',
    type='Markers',
    channel_count=1,
    nominal_srate=0,
    channel_format='int32',
    source_id='id_1')
outlet = StreamOutlet(info)


# 线程中等待连接，按需要
while not exit:
    if outlet.wait_for_consumers(1e-3):
        break
print('Connected')

# 发送
data =123
if outlet.have_consumers():
    outlet.push_sample(data)
    
    
    
## -----------------------------------------------------------------
## 接收数据 ---------------------------------------------------------


lsl_source_id = 'id_1'

if lsl_source_id:
    inlet = True
    streams = resolve_byprop(
        "source_id", lsl_source_id, timeout=5
    )  # Resolve all streams by source_id
    if streams:
        inlet = StreamInlet(streams[0])  # receive stream data

# 读数据
# 不设置timeout 程序将卡在这里
samples, timestamp = inlet.pull_sample(timeout=0)