openlifu
========

|Actions Status| |Documentation Status|

|PyPI version| |PyPI platforms|

|GitHub Discussion|

.. SPHINX-START

``openlifu`` is a toolbox for planning and controlling focused
ultrasound treatments. It generally replicates the functionality
developed in the MATLAB `open-TFUS
toolbox <https://github.com/OpenwaterHealth/opw_neuromod_sw>`__.

Installation
------------

Requirements
~~~~~~~~~~~~

Python 3.10-3.12 on Windows or Linux.

Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Windows:

.. code:: sh

   C:\Users\<uname>\AppData\Local\Programs\Python\Python311\python.exe -m venv env
   .\env\Scripts\activate

Linux:

.. code:: sh

   python3.11 -m venv env

Install project (editable)
~~~~~~~~~~~~~~~~~~~~~~~~~~

With this repo as the working directory:

Basic mode
^^^^^^^^^^
.. code:: sh

   pip install -e .

Dev mode
^^^^^^^^
.. code:: sh

   pip install -e '.[dev]'

Installing Meshroom
-------------------

If you are using ``openlifu.nav.photoscan`` to reconstruct meshes from photo collections, then you will need to set up **Meshroom 2023.3.0**.

Ubuntu
~~~~~~

Download and Extract
^^^^^^^^^^^^^^^^^^^^
1. Download Meshroom 2023.3.0 for Linux from `<https://github.com/alicevision/Meshroom/releases/tag/v2023.3.0>`_.
2. Extract the downloaded archive:

   .. code:: bash

      tar -xvf Meshroom-2023.3.0.tar.gz

Add Meshroom to PATH
^^^^^^^^^^^^^^^^^^^^

**Temporary (Current Session)**
Run:

.. code:: bash

   export PATH="<path-to-meshroom>/Meshroom-2023.3.0:$PATH"

Replace ``<path-to-meshroom>`` with the actual path where Meshroom was extracted.

**Permanent (Persistent Across Sessions)**
For Bash users:

.. code:: bash

   echo 'export PATH="<path-to-meshroom>/Meshroom-2023.3.0:$PATH"' >> ~/.bashrc
   source ~/.bashrc

Windows
~~~~~~~

Download and Extract
^^^^^^^^^^^^^^^^^^^^

1. Download Meshroom 2023.3.0 for Windows from `<https://github.com/alicevision/Meshroom/releases/tag/v2023.3.0>`_.
2. Extract the downloaded archive to a directory of your choice.

Add Meshroom to PATH
^^^^^^^^^^^^^^^^^^^^

1. Open **Edit environment variables for your account** from the Start menu.
2. In the **Environment Variables** window, under **User variables**, select **Path** and click **Edit**.
3. Click **New**, and add the path to the folder containing ``Meshroom.exe``.
4. Click **OK** to save the changes.

Enable GPU Acceleration
^^^^^^^^^^^^^^^^^^^^^^^

To ensure Meshroom uses your NVIDIA GPU:

1. Open **NVIDIA Control Panel**.
2. In the left sidebar under **3D Settings**, click **Manage 3D settings**.
3. Go to the **Program Settings** tab.
4. Click **Add**, then browse to and select ``Meshroom.exe`` from the folder where you extracted Meshroom.
5. Under **Select the preferred graphics processor for this program**, choose **High-performance NVIDIA processor**.
6. Click **Apply**.

Version control of database using DVC (Data Version Control)
-------------------------------------------------------------

Data Version Control (DVC) is a data management tool that is meant to be run alongside Git.
In this project, DVC is used to link changes in the code to specific versions of a sample database containing example project files.
DVC can be used only when this project is installed in Dev mode. Otherwise, it is recommended to use the database that is provided to you. You can read more about DVC and how to use it `here <https://dvc.org/doc/start>`_.

DVC usage
~~~~~~~~~

To download the sample database in Dev mode:

.. code:: sh

   git pull
   dvc remote modify --local shared_gdrive gdrive_client_secret <client_secret_here> # Contact developers for gdrive_client_secret
   dvc pull # Requires access to remote storage

This will download a directory 'db_dvc' in the repo directory that
contains the corresponding version of example database files.

To commit updates to the sample database:

.. code:: sh

   dvc add db_dvc
   git commit -m "Describe updates to database"
   git push
   dvc push #Requires access to remote storage

Disclaimer
----------

CAUTION - Investigational device. Limited by Federal (or United States)
law to investigational use. The system described here has *not* been
evaluated by the FDA and is not designed for the treatment or diagnosis
of any disease. It is provided AS-IS, with no warranties. User assumes
all liability and responsibility for identifying and mitigating risks
associated with using this software.

.. |Actions Status| image:: https://github.com/OpenwaterHealth/OpenLIFU-python/workflows/CI/badge.svg
   :target: https://github.com/OpenwaterHealth/OpenLIFU-python/actions
.. |Documentation Status| image:: https://readthedocs.org/projects/openlifu/badge/?version=latest
   :target: https://openlifu.readthedocs.io/en/latest/?badge=latest
.. |PyPI version| image:: https://img.shields.io/pypi/v/openlifu
   :target: https://pypi.org/project/openlifu/
.. |PyPI platforms| image:: https://img.shields.io/pypi/pyversions/openlifu
   :target: https://pypi.org/project/openlifu/
.. |GitHub Discussion| image:: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
   :target: https://github.com/OpenwaterHealth/OpenLIFU-python/discussions
