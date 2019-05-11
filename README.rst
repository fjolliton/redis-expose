Redis Expose
============

Overview
--------

This tool shows the content of a Redis database.

This can be used to examine the keys and their values by running::

    watch -t -n.5 redis-expose -w auto

Usage
-----

.. code-block:: text

    usage: redis-expose [-h] [-H HOST] [-p PORT] [-d DATABASE] [--no-redir]
                        [-S SECTION] [-s] [-e REGEXP] [-E VALUE_REGEXP] [-i]
                        [-w WIDTH] [-A]

    optional arguments:
      -h, --help            show this help message and exit
      -H HOST, --host HOST  Redis host (default: 127.0.0.1)
      -p PORT, --port PORT  Redis port (default: 6379)
      -d DATABASE, --database DATABASE
                            Database index (default: 0)
      --no-redir            Do not follow MOVED response
      -S SECTION, --section SECTION
                            Show a specific section (info, pubsub, keys, streams)
      -s, --split           Show streams in a separate section. Default is to show
                            them with keys.
      -e REGEXP, --regexp REGEXP
                            Display only keys matching this pattern
      -E VALUE_REGEXP, --value-regexp VALUE_REGEXP
                            Display only keys whose values match this pattern
      -i, --ignore_case     Ignore case when matching key or value
      -w WIDTH, --width WIDTH
                            Limit the maximum width of the output
      -A, --ascii           Use ASCII characters to draw the table

Example
-------

.. code-block:: text

    Redis 5.0.4 (127.0.0.1:6379, 59eaa180),  22 clients,  11 scripts
    Mem: 950.01K used, 4.76M peak, 8.75M rss

    Channel │ Subscribers
    ────────┼────────────
    feeds   │ 17
    reports │ 25
    tasks   │ 3

    Key       │ Type   │ Value
    ──────────┼────────┼─────────────────────────────────────────────────────────────────────
    changelog │ list   │ ['version 0.1: initial development', 'version 0.2: public release']
    events    │ stream │ Info: length=1214, radix-tree-keys=13, radix-tree-nodes=29, groups=0
              │        │ • 1557567280989-0 timestamp=1557567280 text='System reboot'
              │        │ • 1557567299861-0 timestamp=1557567299 text='System ready'
              │        │ • 1557567402302-0 timestamp=1557567402 text='Process A initialized'
              │        │ • 1557567402534-0 timestamp=1557567402 text='Filesystem almost full'
              │        │ ... 1206 more items ...
              │        │ • 1557567406317-0 timestamp=1557567406 text='Connection from 172.17.0.54'
              │        │ • 1557567415358-0 timestamp=1557567415 text='Process C crashed'
              │        │ • 1557567419852-0 timestamp=1557567419 text='Process C started'
              │        │ • 1557567425476-0 timestamp=1557567254 text='System halting'
    info      │ hash   │ {language='Python 3', usage=Console}
    name      │ string │ redis-expose
    scores    │ zset   │ {{Bob=20.0, Carol=40.0, Alice=50.0}}
    version   │ string │ 0.2
