import boto3
import sagemaker
from sagemaker import get_execution_role, pytorch, sklearn, estimator
from functools import partial
import dpath.util as du
import inspect
import sys
import importlib.util
from pathlib import Path
import os
MODULE_EXTENSIONS = '.py'
"""
Proposed usage:

import sagemaker_defaults as sm_defaults

sm_defaults.set_default_vpc (['subnet-123', 'subnet-456'], ['sg-1234567'])
sm_defaults.set_default_s3_key ('arn:aws:kms:my-key-id')
sm_defaults.set_default_role ('arn:aws:iam:role/sagemaker-role')
sm_defaults.set_default_tags ({'tagName': 'tagValue'})

sagemaker_session = sm_defaults.Session ()
...

from sagemaker_defaults import pytorch.PyTorch as PyTorch

pytorch = PyTorch (...)

## Currently supports providing defaults for the following Actions:
  - CreateAutoMlJob
  - CreateProcessingJob
  - CreateTrainingJob

> Note that the following take a user defined number of inputs and as such are not yet supported by this library for default values:
    - CreateAlgorithm
"""

_defaults = {}
_default_session = None


def set_default_vpc(subnet_ids, security_group_ids):
    """
    Set the default VPC configuration in terms of subnet IDs and security group IDs.
    subnet_ids, list of AWS subnet IDs
    security_group_ids, list of AWS security group IDs
    """
    _defaults['vpc_subnet_ids'] = subnet_ids
    _defaults['vpc_security_group_ids'] = security_group_ids


def get_default_vpc_security_group_ids():
    return _defaults.get('vpc_security_group_ids', None)


def get_default_vpc_subnet_ids():
    return _defaults.get('vpc_subnet_ids', None)


def set_default_s3_key(kms_key_id):
    _defaults['s3_key_id'] = kms_key_id


def get_default_s3_key_id():
    return _defaults.get('s3_key_id', None)


def set_default_ebs_key(kms_key_id):
    _defaults['ebs_key_id'] = kms_key_id


def get_default_ebs_key_id():
    return _defaults.get('ebs_key_id', None)


def set_default_tags(tag_dict):
    _defaults['tag_key_values'] = tag_dict


def get_default_tags():
    tag_dict = _defaults.get('tag_key_values', None)
    return [{'Key': k, 'Value': v} for (k, v) in tag_dict.items()]


def set_default_role(role_arn):
    _defaults['partial_role'] = role_arn


def _set_if_absent(d, path, value):
    if '*' in path:
        [pre, post] = path.split('*')
        elem_count = len(du.values(d, f'{pre}*'))
        for i in range(elem_count):
            _set_if_absent(d, f'{pre}{i}{post}', value)
    elif du.search(d, path) == {}:
        du.new(d, path, value())


def _insert_defaults(params, **kwargs):
    action = kwargs['model'].name

    if action in _action_default_map:
        action_defaults = _action_default_map[action]
        for (path, value_func) in action_defaults.items():
            _set_if_absent(params, path, value_func)


def set_session_defaults(sagemaker_session):
    global _default_session

    sm_client = boto3.client('sagemaker')

    event_system = sm_client.meta.events
    event_system.register('provide-client-params.sagemaker', _insert_defaults)

    sagemaker_session.sagemaker_client = sm_client
    if _default_session is None:
        _default_session = sagemaker_session
    return sagemaker_session


def Session(**kwargs):
    session = set_session_defaults(sagemaker.Session(**kwargs))
    return session


def _get_default_role():
    role = get_execution_role()
    if 'partial_role' in _defaults:
        role = _defaults['partial_role']
    return role


def _get_default_session():
    session = _default_session
    if _default_session is None:
        session = Session()
    return session

_action_default_map = {
    'CreateTrainingJob': {
        'RoleArn': _get_default_role,
        'VpcConfig/SecurityGroupIds': get_default_vpc_security_group_ids,
        'VpcConfig/Subnets': get_default_vpc_subnet_ids,
        'OutputDataConfig/KmsKeyId': get_default_s3_key_id,
        'ResourceConfig/VolumeKmsKeyId': get_default_ebs_key_id,
        'Tags': get_default_tags
    },
    'CreateProcessingJob': {
        'RoleArn': _get_default_role,
        'NetworkConfig/VpcConfig/SecurityGroupIds':
        get_default_vpc_security_group_ids,
        'NetworkConfig/VpcConfig/Subnets': get_default_vpc_subnet_ids,
        'ProcessingOutputConfig/KmsKeyId': get_default_s3_key_id,
        'ProcessingResources/ClusterConfig/VolumeKmsKeyId':
        get_default_ebs_key_id,
        'Tags': get_default_tags
    },
    'CreateAutoMLJob': {
        'RoleArn': _get_default_role,
        'AutoMLJobConfig/SecurityConfig/VpcConfig/SecurityGroupIds':
        get_default_vpc_security_group_ids,
        'AutoMLJobConfig/SecurityConfig/VpcConfig/Subnets':
        get_default_vpc_subnet_ids,
        'OutputDataConfig/KmsKeyId': get_default_s3_key_id,
        'AutoMLJobConfig/SecurityConfig/VolumeKmsKeyId':
        get_default_ebs_key_id,
        'Tags': get_default_tags
    }
}

_history = []
_this_m = sys.modules[__name__]

_wrapper_tmpl = """
def {}({}):
    role = _get_default_role()
    if 'sagemaker_session' in locals():
        if sagemaker_session is None:
            sagemaker_session = _get_default_session ()
    return {}(role=role, {})
"""


def _wrap_class(c_name, c_fq_name, c_obj):
    wrapper_sig_args = []
    class_init_args = []

    sig = inspect.signature(c_obj)
    for p_name in sig.parameters:
        if p_name is 'role':
            continue

        p = sig.parameters[p_name]
        if p.name is 'kwargs':
            class_init_args.append('**kwargs')
        else:
            class_init_args.append(p.name + '=' + p.name)
        if p.default is p.empty:
            if p.name is 'kwargs':
                wrapper_sig_args.append('**kwargs')
            else:
                wrapper_sig_args.append(p.name)
        else:
            default_value = str(p.default)
            if isinstance(p.default, str):
                default_value = "'{}'".format(p.default)
            if inspect.isclass(p.default):
                default_value = p.default.__module__ + '.' + p.default.__name__
            wrapper_sig_args.append(p.name + '=' + default_value)

    c_obj_sig = ', '.join(wrapper_sig_args)
    required_arg_str = ', '.join(class_init_args)
    exec(_wrapper_tmpl.format(c_name, c_obj_sig, c_fq_name, required_arg_str))
    return locals()[c_name]


def _wrap_classes(pacmod):
    global _history

    for name, obj in inspect.getmembers(pacmod):
        if inspect.isclass(obj):
            c_name = obj.__module__ + '.' + obj.__name__

            if c_name.startswith('sagemaker.') and c_name not in _history:
                if 'role' in inspect.signature(obj).parameters:
                    _history.append(c_name)
                    if name not in globals(): # don't overwrite statically defined members
                        setattr(_this_m, name, _wrap_class(name, c_name, obj))

        elif inspect.ismodule(obj):
            m_name = obj.__name__

            if m_name.startswith('sagemaker.') and m_name not in _history:
                _history.append(m_name)
                _wrap_classes(obj)


def _list_pkg_modules(pkg_name):
    spec = importlib.util.find_spec(pkg_name)
    if spec is None:
        return set()

    pathname = Path(spec.origin).parent
    ret = set()
    with os.scandir(pathname) as entries:
        for entry in entries:
            if entry.name.startswith('__'):
                continue
            current = '.'.join((pkg_name, entry.name.partition('.')[0]))
            if entry.is_file():
                if entry.name.endswith(MODULE_EXTENSIONS):
                    ret.add(current)
            elif entry.is_dir():
                ret.add(current)
                ret |= _list_pkg_modules(current)

    return ret


for name in _list_pkg_modules('sagemaker'):
    try:
        mod = importlib.import_module(name)
        _wrap_classes(mod)
    except (ModuleNotFoundError, ImportError):
        pass
