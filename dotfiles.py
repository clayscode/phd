#!/usr/bin/env python2.7
from __future__ import print_function

import errno
import inspect
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys

from argparse import ArgumentParser
from collections import deque
from distutils.spawn import find_executable
from time import time

LINUX_DISTROS = ['ubuntu']

def get_platform():
    distro = platform.linux_distribution()
    if not distro[0]:
        return {
            "darwin": "osx",
        }.get(sys.platform, sys.platform)
    else:
        return distro[0].lower()

PLATFORM = get_platform()
HOSTNAME = socket.gethostname()
DOTFILES = os.path.expanduser("~/.dotfiles")
PRIVATE = os.path.expanduser("~/Dropbox/Shared")


class Colors:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


class Task(object):
    """
    A Task is a unit of work.

    Each task consists of a run() method

    {setup,run,teardown}_<platform>()

    Attributes:
        __platforms__ (List[str], optional): A list of platforms which the
            task may be run on. Any platform not in this list will not
            execute the task.
        __deps__ (List[Task], optional): A list of tasks which must be executed
            before this task may be run.
        __genfiles__ (List[str], optional): A list of permanent files generated
            during execution of this task.
        __tmpfiles__ (List[str], optional): A list of files generated during
            execution of this task.

    Methods:
        setup():
        run():
        teardown():
        run_osx()
    run()
    """

    def setup(self):
        pass

    def run(self):
        """ """
        raise NotImplementedError

    def teardown(self):
        pass

    def __repr__(self):
        return type(self).__name__


class InvalidTaskError(Exception): pass


def is_compatible(a, b):
    """ return if platforms a and b are compatible """
    if b == "linux":
        return a in LINUX_DISTROS or a == "linux"
    else:
        return a == b


def which(binary):
    return find_executable(binary)


def mkdir(path):
    path = os.path.abspath(os.path.expanduser(path))
    logging.debug("$ mkdir " + path)
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def _shell(*args, **kwargs):
    error = kwargs.get("error", True)
    stdout = kwargs.get("stdout", False)

    logging.debug("$ " + "".join(*args))
    if stdout:
        return subprocess.check_output(*args, shell=shell, universal_newlines=True,
                                       stderr=subprocess.PIPE)
    elif error:
        return subprocess.check_call(*args, shell=shell, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
    else:
        try:
            subprocess.check_call(*args, shell=shell, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            return False

def shell(*args):
    return _shell(*args)


def shell_ok(*args):
    return _shell(*args, error=False)


def shell_output(*args):
    return _shell(*args, stdout=True)


def symlink(src, dst, sudo=False):
    src = os.path.expanduser(src)
    dst = os.path.expanduser(dst)

    if src.startswith("/"):
        src_abs = src
    else:
        src_abs = os.path.dirname(dst) + "/" + src

    # Symlink already exists
    use_sudo = "sudo -H " if sudo else ""
    if shell_ok("{use_sudo}test -f '{dst}'".format(**vars())):
        linkdest = shell_output("{use_sudo}readlink {dst}".format(**vars())).rstrip()
        if linkdest.startswith("/"):
            linkdest_abs = linkdest
        else:
            linkdest_abs = os.path.dirname(dst) + "/" + linkdest
        if linkdest_abs == src_abs:
            return

    if not (shell_ok("{use_sudo}test -f '{src_abs}'".format(**vars())) or
            shell_ok("{use_sudo}test -d '{src_abs}'".format(**vars()))):
        raise OSError("symlink source '{src}' does not exist".format(**vars()))
    # if shell_ok("{use_sudo}test -d '{dst}'".format(**vars())):
    #     raise OSError("symlink destination '{dst}' is a directory".format(**vars()))

    # Make a backup of existing file:
    if shell_ok("{use_sudo}test -f '{dst}'".format(**vars())):
        shell("{use_sudo}mv {dst} {dst}.backup".format(**vars()))

    # Create the symlink:
    shell("{use_sudo}ln -s {src} {dst}".format(**vars()))


def copy(src, dst):
    src = os.path.expanduser(src)
    dst = os.path.expanduser(dst)

    if not os.path.isfile(src):
        raise OSError("copy source '{src}' does not exist".format(**vars()))
    if os.path.isdir(dst):
        raise OSError("copy destination '{dst}' is a directory".format(**vars()))

    logging.debug("$ cp '{src}' '{dst}'".format(**vars()))
    shutil.copyfile(src, dst)


def clone_git_repo(url, destination, version):
    destination = os.path.abspath(os.path.expanduser(destination))

    # clone repo if necessary
    if not os.path.isdir(destination):
        shell('git clone --recursive "{url}" "{destination}"'.format(**vars()))

    if not os.path.isdir(os.path.join(destination, ".git")):
        raise OSError('directory "' + os.path.join(destination, ".git") +
                      '" does not exist')

    # set revision
    pwd = os.getcwd()
    os.chdir(destination)
    target_hash = shell_output("git rev-parse {version} 2>/dev/null".format(**vars()))
    current_hash = shell_output("git rev-parse HEAD".format(**vars()))

    if current_hash != target_hash:
        logging.info("setting repo version {destination} to {version}".format(**vars()))
        shell("git fetch --all")
        shell("git reset --hard '{version}'".format(**vars()))

    os.chdir(pwd)


def is_runnable_task(obj):
    """ returns true if object is a task for the current platform """
    if not (inspect.isclass(obj) and  issubclass(obj, Task) and obj != Task):
        return False

    task = obj()
    platforms = getattr(task, "__platforms__", [])
    if not any(is_compatible(PLATFORM, x) for x in platforms):
        logging.debug("skipping " + type(task).__name__ + " on platform " + PLATFORM)
        return False

    return True


# must be imported before get_tasks()
from tasks import *


def get_tasks():
    """ generate the list of tasks to run, respecting dependencies """
    tasks = [x[1]() for x in inspect.getmembers(sys.modules[__name__], is_runnable_task)]
    tasks = sorted(tasks, key=lambda task: type(task).__name__)

    queue = deque(tasks)
    ordered = list()
    while len(queue):
        task = queue.popleft()
        deps = getattr(task, "__deps__", [])
        deps += getattr(task, "__" + PLATFORM + "_deps__", [])

        for dep in deps:
            if dep not in set(type(task) for task in ordered):
                queue.append(task)
                break
        else:
            ordered.append(task)

    return ordered


def get_task_method(task, method_name):
    """ resolve task method. First try and match <method>_<platform>(),
        then <method>() """
    fn = getattr(task, method_name + "_" + PLATFORM, None)
    if fn is None and PLATFORM in LINUX_DISTROS:
        fn = getattr(task, method_name + "_linux", None)
    if fn is None:
        fn = getattr(task, method_name, None)
    if fn is None:
        raise InvalidTaskError
    return fn


def main(*args):
    """ main dotfiles method """

    # Parse arguments
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--verbose', action='store_true')
    group.add_argument('--debug', action='store_true')
    args = parser.parse_args(args)

    # Configure logger
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    logging.basicConfig(format="%(message)s")

    # Get the list of tasks to run
    logging.debug("creating tasks list ...")
    queue = get_tasks()
    done = set()
    ntasks = len(queue)
    print("Running " + Colors.BOLD + str(ntasks) + Colors.END +
          " tasks on", PLATFORM + ":")

    # Run the tasks
    errored = False
    try:
        for i, task in enumerate(queue):
            # Task setup
            logging.debug(type(task).__name__ + " setup")

            # Resolve and run setup() method:
            get_task_method(task, "setup")()

            done.add(task)

            # Resolve and run run() method:
            print("  [{:2d}/{:2d}]".format(i + 1, ntasks) + Colors.BOLD,
                  type(task).__name__, Colors.END + "...", end=" ")
            if logging.getLogger().level <= logging.INFO:
                print()
            sys.stdout.flush()
            start_time = time()
            get_task_method(task, "run")()
            runtime = time() - start_time

            print("{:.3f}s".format(runtime))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\ninterrupt")
        errored = True
    except Exception as e:
        print(Colors.BOLD + Colors.RED + type(e).__name__)
        print(e, Colors.END)
        errored = True
        if logging.getLogger().level <= logging.DEBUG:
            raise
    finally:
        # Task teardowm
        logging.debug("")
        for task in done:
            logging.debug("  " + Colors.BOLD + type(task).__name__ + " teardown" + Colors.END)
            get_task_method(task, "teardown")()

            # remove any temporary files
            for file in getattr(task, "__tmpfiles__", []):
                file = os.path.abspath(os.path.expanduser(file))
                if os.path.exists(file):
                    logging.debug("rm {file}".format(**vars()))
                    os.remove(file)

    if not errored:
        print("done")
    return 1 if errored else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
