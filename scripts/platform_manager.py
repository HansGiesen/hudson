#!/usr/bin/env python3

import argparse
import logging
import os
import re
import socketserver
import subprocess
import threading
import time
import yaml


def acquire(client, platform):
    with condition:
        queue = wait_queue[platform]
        thread = threading.current_thread()
        queue.append(thread)
        while queue[0] != thread or len(available[platform]) == 0:
            condition.wait()
        queue.pop(0)
        host = available[platform].pop(0)
        return host

def release(client, platform, host):
    with condition:
        logging.info('%s released %s.', client, host)
        available[platform].append(host)
        condition.notify_all()

def reboot(host):
    return subprocess.call('ssh -o ConnectTimeout=10 ' + host + ' /sbin/reboot',
                           shell=True,
                           stderr=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL) == 0

def ping(host):
    result = subprocess.run('ssh -o ConnectTimeout=10 ' + host + ' "echo OK"',
                            shell=True,
                            stderr=subprocess.DEVNULL,
                            stdout=subprocess.PIPE)
    return result.returncode == 0 and result.stdout == b'OK\n'

def log_temperatures(platform, host):
    if platform != 'zcu102':
        return
    template = '/sys/bus/iio/devices/iio:device0/in_temp*_%s_temp_raw'
    sensors = ['ps', 'pl', 'remote']
    files = ' '.join([template % sensor for sensor in sensors])
    cmd = "bash -c 'cat %s'" % files
    cmd = 'ssh -o ConnectTimeout=10 %s "%s"' % (host, cmd) 
    result = subprocess.run(cmd,
                            shell=True,
                            stderr=subprocess.DEVNULL,
                            stdout=subprocess.PIPE)
    if result.returncode != 0:
        return
    values = [int(value) for value in result.stdout.decode().split()]
    if len(values) != len(sensors):
        return
    values = [509.314 * value / 65536.0 - 280.23 for value in values]
    text = ', '.join('%s = %.1f' % (sensor, value)
                     for sensor, value in zip(sensors, values))
    logging.info('Temperatures of %s: %s', host, text)

def send_mail(msg):
    email_addr = cfg['email_addr']
    cmd = 'echo "%s" | mail -s "HuDSoN alert" %s' % (msg, email_addr)
    subprocess.run(cmd,
                   shell=True,
                   stderr=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL)

def prepare(client, platform):
    while True:
        host = acquire(client, platform)
        logging.info('Rebooting %s.', host)
        if reboot(host):
            logging.info('Pinging %s.', host)
            success = False
            for attempt in range(6):
                time.sleep(10)
                success = ping(host)
                if success:
                    break
            if success:
                logging.info('%s acquired %s.', client, host)
                break
        logging.info('%s has gone offline.', host)
        send_mail('%s has gone offline.' % host)
        with condition:
            offline[host] = platform
    log_temperatures(platform, host)
    return host

def ping_thread():
    while True:
        time.sleep(900.0)
        for host, platform in list(offline.items()):
            if ping(host):
                logging.info('%s has come online.', host)
                send_mail('%s has come online.' % host)
                with condition:
                    available[platform].append(host)
                    del offline[host]
                    condition.notify_all()

class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        match = self.process_request(r'client (\S+)\n')
        if match is None:
            return
        client = match.group(1)
        logging.info('%s has connected.', client)
        while True:
            match = self.process_request(r'acquire (\S+)\n')
            if match is None:
                break
            platform = match.group(1)
            if platform not in available:
                response = 'Invalid platform.\n'
                self.wfile.write(response.encode('utf-8'))
                continue
            host = prepare(client, platform)
            response = (host + '\n').encode('utf-8')
            self.wfile.write(response)
            match = self.process_request(r'release\n')
            release(client, platform, host)
            if match is None:
                break
        logging.info('%s has disconnected.', client)
    def process_request(self, pattern):
        while True:
            data = self.rfile.readline()
            if not data:
                return None
            request = data.decode('utf-8')
            match = re.match(pattern, request)
            if match:
                return match
            response = 'Invalid command.\n'
            self.wfile.write(response.encode('utf-8'))

class TCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


parser = argparse.ArgumentParser(description='Share platforms among tuners.')
parser.add_argument('--log', help='Log file')

args = parser.parse_args()

kvargs = {'level': logging.INFO, 'format': '%(asctime)s %(message)s'}
if args.log:
    kvargs['filename'] = args.log
logging.basicConfig(**kvargs)

script_dir = os.path.dirname(os.path.realpath(__file__))
cfg_path = os.path.join(script_dir, '../cfg.yml')
with open(cfg_path, 'r') as cfg_file:
    data = cfg_file.read()
cfg = yaml.safe_load(data)

available = {}
wait_queue = {}
offline = {}
platforms = cfg['platforms']['instances']
for host in platforms:
    platform = host['type']
    hostname = host['hostname']
    queue = available.setdefault(platform, [])
    queue.append(hostname)
    available[platform] = queue
    wait_queue[platform] = []

condition = threading.Condition()

thread = threading.Thread(target=ping_thread, daemon=True)
thread.start()

port = cfg['platform_manager']['port']
with TCPServer(('localhost', port), RequestHandler) as server:
    server.serve_forever()
