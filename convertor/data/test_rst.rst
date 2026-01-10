.. _api-reference:

=================================
API Reference: Data Processing
=================================

.. module:: dataprocessor
   :synopsis: High-performance data processing utilities with modern algorithms

.. moduleauthor:: Jane Smith <jane@example.com>

Overview
========

.. highlights::

   * Fast parallel processing with configurable workers
   * Memory-efficient streaming operations
   * Extensive format support (CSV, JSON, Parquet, HDF5)
   * Built-in error handling and validation

Installation
------------

.. code-block:: bash

   pip install data processor
   # Or for development version
   pip install git+https://github.com/example/dataprocessor.git

Quick Start
-----------

.. code-block:: python
   :linenos:
   :emphasize-lines: 3,5

   from dataprocessor import Pipeline

   pipeline = Pipeline()
   pipeline.add_step('normalize')
   result = pipeline.run(data)

Classes
=======

Pipeline
--------

.. class:: Pipeline(config=None, workers=4)

   Main data processing pipeline class with chainable operations.

   :param config: Configuration dictionary for pipeline settings
   :type config: dict, optional
   :param workers: Number of parallel workers for concurrent processing
   :type workers: int

   .. attribute:: steps
      :type: list

      List of processing steps in the pipeline, executed in order.

   .. method:: add_step(name, **kwargs)

      Add a processing step to the pipeline.

      :param name: Step identifier (normalize, filter, aggregate, etc.)
      :type name: str
      :param kwargs: Step-specific parameters
      :returns: Self for method chaining
      :rtype: Pipeline
      :raises ValueError: If step name is unknown or invalid parameters

      .. rubric:: Example

      .. code-block:: python

         pipeline.add_step('normalize', method='zscore')
         pipeline.add_step('filter', threshold=0.5)

   .. method:: run(data)

      Execute the pipeline on input data.

      :param data: Input data to process
      :type data: numpy.ndarray or pandas.DataFrame
      :returns: Processed data with transformations applied
      :rtype: numpy.ndarray

      .. warning::

         Large datasets (>1GB) may require significant memory.
         Consider using batch processing for very large files.

Functions
=========

.. function:: process_batch(items, callback=None)

   Process multiple items in batch mode with optional progress tracking.

   :param items: Items or records to process
   :type items: Iterable
   :param callback: Progress callback function called after each item
   :type callback: callable, optional

   .. deprecated:: 2.0
      Use :meth:`Pipeline.run_batch` instead for better performance.

Constants
=========

.. data:: DEFAULT_WORKERS
   :value: 4

   Default number of parallel workers for processing operations.

.. data:: SUPPORTED_FORMATS

   List of supported input and output formats.

   .. code-block:: python

      SUPPORTED_FORMATS = ['csv', 'json', 'parquet', 'hdf5', 'arrow']

Exceptions
==========

.. exception:: ProcessingError

   Raised when a processing step fails or encounters  invalid data.

   .. attribute:: step_name

      Name of the pipeline step where the error occurred.

   .. attribute:: original_error

      The underlying exception that triggered this error.

Notes and Warnings
==================

.. note::

   This module requires Python 3.8 or higher for optimal performance.

.. warning::

   Memory usage scales linearly with dataset size. Monitor system resources
   when processing datasets larger than available RAM.

.. danger::

   Never use in production without proper error handling and input validation.
   Malformed data can cause unexpected behavior.

.. tip::

   Use `workers=1` for debugging to get clearer error messages and stack traces.

.. seealso::

   * :doc:`/tutorials/getting-started`
   * :doc:`/guides/performance-tuning`
   * :doc:`/guides/error-handling`
   * `NumPy Documentation <https://numpy.org/doc/>`_

Tables
======

.. list-table:: Supported Operations
   :header-rows: 1
   :widths: 20 20 60

   * - Operation
     - Complexity
     - Description
   * - normalize
     - O(n)
     - Scale values to specified range or standardize
   * - filter
     - O(n)
     - Remove values below threshold
   * - aggregate
     - O(n log n)
     - Group and summarize data
   * - transform
     - O(n)
     - Apply custom transformation function

Version History
===============

.. versionadded:: 1.0
   Initial release with basic pipeline functionality.

.. versionchanged:: 1.5
   Added parallel processing support with configurable workers.

.. versionchanged:: 2.0
   Complete API redesign for better performance and usability.
   Breaking changes include removal of legacy batch processing methods.

Index and References
====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installation
   quickstart
   api/index
   examples/index
   changelog
