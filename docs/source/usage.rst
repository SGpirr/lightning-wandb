Usage
=====

.. _installation:

Installation
------------
아나콘다 환경 생성
.. code-block:: console
    export CONDA_ENV=lightning-wandb
    conda create -n $CONDA_ENV python=3.9 -y
    conda activate $CONDA_ENV
    conda install nb_conda_kernels -y
    pip install --upgrade pip
    pip install ipykernel
    python -m ipykernel install --user --name $CONDA_ENV --display-name $CONDA_ENV

(git으로 추적하지 않을) 디렉터리 생성
이 부분은 템플릿 레포지토리로 대체할 예정
.. code-block:: console
    bash create_dirs.sh
    
이 밑의 부분은 그냥 남겨둘게요

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

