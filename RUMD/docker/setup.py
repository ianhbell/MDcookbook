import os
import sys
import glob
import subprocess
import multiprocessing

from setuptools import setup
from distutils.cmd import Command
from distutils.command.build import build
from distutils.command.install import install

class rumd_build_cmd(build):
    def run(self):
        # build rumd
        cmd = ['make']

        try:
            cmd.append('-j%d' % multiprocessing.cpu_count())
        except NotImplementedError:
            print('Unable to determine number of CPUs. Using single threaded make.')

        def compile():
            subprocess.call(cmd)

        self.execute(compile, [], "Compiling rumd")
        # I don't see why there's an extra layer of indirection here
        # why not: self.execute(subprcoess.call, [cmd], "Compiling rumd") ??

        # run original build code
        build.run(self)

        # Copy the built files into the appropriate location in build/lib
        for filename in glob.glob('Python/rumd/*.so'):
            self.move_file(filename, self.build_lib)

class PostInstallCommand(install):
    """
    Post-installation step of copying built executables and shared libraries to appropriate location

    Inspiration from https://github.com/benfred/py-spy/blob/290584dde76834599d66d74b64165dfe9a357ef5/setup.py#L42
    """
    def run(self):
        """
        Overload of run command from distutils class
        """
        install.run(self)

        # First make sure the scripts directory exists
        if not os.path.isdir(self.install_scripts):
            os.makedirs(self.install_scripts)

        target = self.install_scripts
        if os.path.isfile(target):
            os.remove(target)

        for filename in glob.glob('Tools/*'):
            if os.access(filename, os.X_OK):
                self.move_file(filename, self.install_scripts)

class rumd_test_cmd(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        retcode = subprocess.call(['make', 'test'])

setup(
    name='rumd',
    author='Nicholas Bailey',
    author_email='nbailey@ruc.dk',
    url='http://rumd.org',
    version='3.5',
    description='C++/CUDA-based molecular dynamics simulation code for nVidia GPUs',
    packages=['rumd'],
    package_dir={'':'Python'},
    classifiers=['Programming Language :: Python',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: C++',
                 'Programming Language :: CUDA',
                 'Operating System :: Unix',
                 'Intended Audience :: Science/Research'],
    cmdclass={'build': rumd_build_cmd,
              'install': PostInstallCommand,
              'test': rumd_test_cmd},
)
