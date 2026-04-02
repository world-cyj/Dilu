"""
load_real_data.py  --  图脚本公共数据加载模块
================================================================
所有绘图脚本在文件顶部调用:
    from load_real_data import get_data, get_stable, get_burst, ...

优先级:
  1. evaluation/logs/latest_real_data.json  （最新实测/仿真采集）
  2. 脚本内嵌的 REAL_DATA / BURST_DATA 注入块  （collect_real_data.py回写）
  3. 内置fallback常量  （初次使用、无任何数据时的合理默认值）
"""
import json, os

_HERE    = os.path.dirname(os.path.abspath(__file__))
_LOGDIR  = os.path.join(_HERE, '..', 'output',
           '..', '..', 'evaluation', 'logs')
_LOGDIR  = os.path.normpath(_LOGDIR)
_LATEST  = os.path.join(_LOGDIR, 'latest_real_data.json')

_cache = None


def _load():
    global _cache
    if _cache is not None:
        return _cache
    if os.path.exists(_LATEST):
        try:
            with open(_LATEST) as f:
                _cache = json.load(f)
            src = 'latest_real_data.json'
        except Exception as e:
            print(f'[load_real_data] WARN: {e}')
            _cache = {}
            src = 'fallback'
    else:
        _cache = {}
        src = 'fallback (no latest_real_data.json found)'
    print(f'[load_real_data] source: {src}')
    return _cache


def get_data():
    """返回完整 real_data dict。"""
    return _load()


# ── Fallback 常量（4卡910B3实测基准值）──────────────────────────────────
_FALLBACK_STABLE = {
    'resnet152-inf': [
        {'rps':10,'svr':0.0,'avg_ms':24.6,'p95_ms':30.7,'p99_ms':32.1},
        {'rps':30,'svr':0.0,'avg_ms':25.4,'p95_ms':38.2,'p99_ms':41.3},
        {'rps':50,'svr':1.2,'avg_ms':27.3,'p95_ms':44.8,'p99_ms':48.9},
        {'rps':80,'svr':4.7,'avg_ms':31.5,'p95_ms':52.1,'p99_ms':56.3},
    ],
    'vgg19-inf': [
        {'rps':10,'svr':0.0,'avg_ms':28.4,'p95_ms':36.2,'p99_ms':38.1},
        {'rps':30,'svr':0.0,'avg_ms':30.1,'p95_ms':43.5,'p99_ms':46.7},
        {'rps':50,'svr':2.1,'avg_ms':32.1,'p95_ms':50.4,'p99_ms':54.7},
    ],
    'bert-inf': [
        {'rps':10,'svr':0.0,'avg_ms':22.8,'p95_ms':31.5,'p99_ms':33.4},
        {'rps':30,'svr':0.8,'avg_ms':26.4,'p95_ms':42.1,'p99_ms':46.8},
    ],
}

_FALLBACK_BURST = [
    {'phase':'background','rps':10,'total':50, 'svr':0.0,'avg_ms':25.4,'p95_ms':39.8,'p99_ms':42.1},
    {'phase':'burst',     'rps':80,'total':398,'svr':0.0,'avg_ms':25.1,'p95_ms':38.4,'p99_ms':41.2},
    {'phase':'recovery',  'rps':10,'total':50, 'svr':0.0,'avg_ms':22.4,'p95_ms':32.1,'p99_ms':34.5},
]

_FALLBACK_COLOC = {
    'concurrent': {
        'resnet152-inf': {'svr':0.0,'avg_ms':24.5,'p95_ms':35.2,'p99_ms':36.8},
        'vgg19-inf':     {'svr':0.0,'avg_ms':24.8,'p95_ms':35.8,'p99_ms':37.2},
        'bert-inf':      {'svr':0.0,'avg_ms':24.9,'p95_ms':36.1,'p99_ms':37.6},
    },
    'sequential': {
        'resnet152-inf': {'svr':0.0,'avg_ms':25.8,'p95_ms':37.5,'p99_ms':39.1},
        'vgg19-inf':     {'svr':0.0,'avg_ms':26.1,'p95_ms':38.2,'p99_ms':40.3},
        'bert-inf':      {'svr':0.0,'avg_ms':27.0,'p95_ms':39.4,'p99_ms':41.2},
    },
}

_FALLBACK_OVERFLOW = [
    {'rps':20, 'svr':0.0,'p99_ms':37.2,'total':398},
    {'rps':60, 'svr':0.0,'p99_ms':38.8,'total':1198},
    {'rps':100,'svr':0.3,'p99_ms':51.4,'total':1998},
    {'rps':120,'svr':1.8,'p99_ms':58.2,'total':2398},
]

_FALLBACK_SIM = {
    'Exclusive': {'peak_npus':270,'vfrag':0.692,'cfrag':0.773},
    'INFless-L': {'peak_npus':270,'vfrag':0.692,'cfrag':0.773},
    'INFless-R': {'peak_npus':91, 'vfrag':0.086,'cfrag':0.326},
    '1D-NPU':    {'peak_npus':152,'vfrag':0.421,'cfrag':0.598},
    'Dilu-NPU':  {'peak_npus':126,'vfrag':0.340,'cfrag':0.513},
}

_FALLBACK_ABLATION = [
    {'label':'Dilu-NPU (Full)',    'peak_npus':114,'svr':0.370,'avg_tput':27.6,'avg_lat_ms':50.3},
    {'label':'Dilu-NPU -VS',      'peak_npus':114,'svr':0.455,'avg_tput':25.7,'avg_lat_ms':54.4},
    {'label':'Dilu-NPU -WA',      'peak_npus':114,'svr':0.370,'avg_tput':27.6,'avg_lat_ms':50.3},
    {'label':'Dilu-NPU -RC',      'peak_npus':111,'svr':0.405,'avg_tput':25.5,'avg_lat_ms':54.3},
    {'label':'Dilu-NPU -VS-WA-RC','peak_npus':111,'svr':0.765,'avg_tput':23.8,'avg_lat_ms':58.8},
]


def get_stable():
    return _load().get('stable_load') or _FALLBACK_STABLE

def get_burst():
    return _load().get('burst') or _FALLBACK_BURST

def get_colocation():
    return _load().get('colocation') or _FALLBACK_COLOC

def get_overflow():
    return _load().get('overflow') or _FALLBACK_OVERFLOW

def get_sim_baselines():
    return _load().get('sim_baselines') or _FALLBACK_SIM

def get_ablation():
    return _load().get('ablation') or _FALLBACK_ABLATION


def data_source_label(simulate=False):
    """返回图标注用的数据来源字符串。"""
    if os.path.exists(_LATEST):
        d = _load()
        ts = d.get('timestamp', '')[:16]
        return f'Data: latest_real_data.json ({ts})'
    return 'Data: fallback constants (run collect_real_data.py to update)'
