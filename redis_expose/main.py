# Redis Expose
# Copyright (C) 2019  Frédéric Jolliton
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# [ ] Better format stream (not like other keys)

import shutil
import argparse
import re
import json

from unicodedata import category
from itertools import zip_longest
from typing import Optional, Any, Iterable, Tuple, List, Set, Callable, Union, NamedTuple, Pattern

import redis
import redis.exceptions


SECTIONS = ['info', 'pubsub', 'keys', 'streams']

# Hard limit when decoding string
STRING_LIMIT = 500


def section(value: str) -> str:
    """
    Parser for the section argument
    """
    if value not in SECTIONS:
        raise argparse.ArgumentTypeError(f'no such section {value!r}')
    return value


def width(value: str) -> Union[str, int]:
    """
    Parser for the width argument
    """
    if value == 'auto':
        return value
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('expected a non negative value')
    return value


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 argument_default=argparse.SUPPRESS)
parser.add_argument('-H', '--host', default='127.0.0.1', help='Redis host')
parser.add_argument('-p', '--port', type=int, default=6379, help='Redis port')
parser.add_argument('-d', '--database', type=int, default=0, help='Database index')
parser.add_argument('--no-redir', action='store_true',
                    help='Do not follow MOVED response')
parser.add_argument('-S', '--section', type=section, action='append',
                    help=f'Show a specific section ({", ".join(SECTIONS)})')
parser.add_argument('-s', '--split', action='store_true',
                    help='Show streams in a separate section. Default is to show them with keys.')
parser.add_argument('-e', '--regexp', action='append',
                    help='Display only keys matching this pattern')
parser.add_argument('-E', '--value-regexp', action='append',
                    help='Display only keys whose values match this pattern')
parser.add_argument('--and', action='store_true',
                    help='Both -e and -E must match. Default is to require only either -e or -E to match.')
parser.add_argument('-i', '--ignore_case', action='store_true',
                    help='Ignore case when matching key or value')
parser.add_argument('-w', '--width', type=width,
                    help='Limit the maximum width of the output')
parser.add_argument('-A', '--ascii', action='store_true',
                    help='Use ASCII characters to draw the table')


def table(rows: Iterable[Iterable[Any]], *, write=print, header: Optional[int] = 0,
          ascii_only: bool = False, hide_columns: Optional[Set] = None) -> None:
    """
    Format a list of rows as a table

    :param rows: the rows to format, as a list of lists. Elements
      should be either string, or a list of strings for multiline
      cells.
    :param header: the number of rows to consider as header.
    :param ascii_only: use only ASCII characters for drawing the
      table.
    :param formatter: a function to convert element to bytestring.

    >>> table([['Name', 'Score', 'Games'],
    ...        ['Alice', '2000', ['foo', 'bar']],
    ...        ['Bob', '750', 'baz']],
    ...       header=1, ascii_only=True)
    Name  | Score | Games
    ------+-------+------
    Alice | 2000  | foo
          |       | bar
    Bob   | 750   | baz
    """
    if ascii_only:
        separators = [' | ', '-+-', '-']
    else:
        separators = [' \N{BOX DRAWINGS LIGHT VERTICAL} ',
                      '\N{BOX DRAWINGS LIGHT HORIZONTAL}'
                      '\N{BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL}'
                      '\N{BOX DRAWINGS LIGHT HORIZONTAL}',
                      '\N{BOX DRAWINGS LIGHT HORIZONTAL}']

    header = max(0, header or 0)

    if hide_columns is None:
        hide_columns = set()

    # Expand multiline cells
    final_rows: List[List[str]] = []
    raw_header = -1
    for i, row in enumerate(rows):
        final_rows += list(zip_longest(*(cell if isinstance(cell, list) else [cell]
                                         for i, cell in enumerate(row)
                                         if i not in hide_columns),
                                       fillvalue=''))
        if i == header - 1:
            raw_header = len(final_rows)

    # Compute the maximum width of each columns.
    columns: List[int] = []
    for row in final_rows:
        columns += [0] * max(0, len(row) - len(columns))
        for i, item in enumerate(row):
            columns[i] = max(columns[i], len(item))

    # Print the table
    for j, row in enumerate(final_rows):
        line = separators[0].join(item.ljust(width) for item, width in zip(row, columns))
        write(line.rstrip())
        if j == raw_header - 1:
            line = separators[1].join(separators[2]*width for width in columns)
            write(line)


RE_SAFE_STRING = re.compile(r'''^(?:(?![='"\[\]{}()])[\x21-\x7e])+$''')

RE_UNQUOTE = re.compile(r'\\u([0-9a-f]{4})')


def quote(s: str) -> str:
    r"""
    Quote a string using JSON

    :param s: the string to quote

    >>> print(quote('Ùñįçòḑē\n\x01'))
    "Ùñįçòḑē\n\u0001"
    """
    s = json.dumps(s)
    def dec(r):
        char = chr(int(r.group(1), 16))
        cat = category(char)
        return char if cat not in ('Zl', 'Zp', 'Cc', 'Cf', 'Cs', 'Co', 'Cn') else r.group(0)
    return RE_UNQUOTE.sub(dec, s)


def decode_redis(s: bytes, *, simple=False, as_hex=False) -> str:
    r"""
    Attempt to decode string as UTF-8

    If it fails, return a hexadecimal representation.

    :param s: the bytes to decode
    :param simple: if `True`, then do not quote the result if it
      contains "safe" characters
    :param as_hex: if `True`, always format with hexadecimal
      representation

    >>> print(decode_redis(b'hit:index'))
    "hit:index"
    >>> print(decode_redis(b'hit:index', simple=True))
    hit:index
    >>> print(decode_redis(b'\xde\xad\xbe\xef'))
    <<de ad be ef>>
    """
    if len(s) > STRING_LIMIT:
        s = s[:STRING_LIMIT]
        extra = '...'
    else:
        extra = ''
    def format_as_hex():
        r = ' '.join(f'{b:02x}' for b in s)
        return f"<<{r}{extra}>>"
    if as_hex:
        return format_as_hex()
    try:
        text = s.decode()
    except UnicodeDecodeError:
        return format_as_hex()
    else:
        if simple and not extra and RE_SAFE_STRING.match(text):
            return text
        else:
            return quote(text) + extra


class Key(NamedTuple):
    name: str
    value: List[str]
    type: str
    ttl: Optional[int]
    redir: Optional[str]


def fetch_string(client: redis.Redis, key: bytes, *, ascii_only=False) -> List[str]:
    """
    Fetch the details about a string

    :param client: the Redis connection
    :param key: the stream to examine
    :param ascii_only: if `True`, use only ASCII characters for
      decorations
    """
    value = client.get(key)
    return [decode_redis(value, simple=True)] if value is not None else []


def fetch_list(client: redis.Redis, key: bytes, *, ascii_only=False) -> List[str]:
    """
    Fetch the details about a list

    :param client: the Redis connection
    :param key: the stream to examine
    :param ascii_only: if `True`, use only ASCII characters for
      decorations
    """
    items = [decode_redis(v, simple=True) for v in client.lrange(key, 0, -1)]
    return ['[' + ', '.join(items) + ']']


def fetch_hash(client: redis.Redis, key: bytes, *, ascii_only=False) -> List[str]:
    """
    Fetch the details about a hash

    :param client: the Redis connection
    :param key: the stream to examine
    :param ascii_only: if `True`, use only ASCII characters for
      decorations
    """
    hkeys = client.hkeys(key)
    if not hkeys:
        return []
    items = zip([decode_redis(k, simple=True) for k in hkeys],
                [decode_redis(v, simple=True) if v is not None else '(null)' for v in client.hmget(key, *hkeys)])
    return ['{' + ', '.join(f'{k}={v}' for k, v in items) + '}']


def fetch_set(client: redis.Redis, key: bytes, *, ascii_only=False) -> List[str]:
    """
    Fetch the details about a set

    :param client: the Redis connection
    :param key: the stream to examine
    :param ascii_only: if `True`, use only ASCII characters for
      decorations
    """
    members = client.smembers(key)
    if members:
        return ['{' + ', '.join(decode_redis(member, simple=True) for member in sorted(members)) + '}']
    else:
        return []


def fetch_zset(client: redis.Redis, key: bytes, *, ascii_only=False) -> List[str]:
    """
    Fetch the details about a sorted set

    :param client: the Redis connection
    :param key: the stream to examine
    :param ascii_only: if `True`, use only ASCII characters for
      decorations
    """
    members = dict(client.zrange(key, 0, -1, withscores=True))
    return ['{{' + ', '.join(f'{decode_redis(k, simple=True)}: {v}' for k, v in members.items()) + '}}']


def fetch_stream(client: redis.Redis, key: bytes, *, ascii_only=False) -> List[str]:
    """
    Fetch the details about a stream

    :param client: the Redis connection
    :param key: the stream to examine
    :param ascii_only: if `True`, use only ASCII characters for
      decorations
    """
    context = 4

    with client.pipeline() as pipe:
        pipe.multi()
        pipe.xinfo_groups(key)
        pipe.xinfo_stream(key)
        pipe.xlen(key)
        pipe.xrange(key, count=context)
        pipe.xrevrange(key, count=context)
        groups, streams, total, firsts, lasts = pipe.execute()
        lasts = lasts[::-1]

    consumers = {}
    for group in groups:
        consumers[group['name']] = client.xinfo_consumers(key, group['name'])

    result = []

    # Info

    def dec(v):
        return decode_redis(v, simple=True) if isinstance(v, bytes) else str(v)

    interesting_keys = {'length', 'radix-tree-keys', 'radix-tree-nodes', 'groups'}

    result.append('Info: ' + ', '.join(f'{k}={dec(v)}' for k, v in streams.items() if k in interesting_keys))

    # Groups

    for group in groups:
        rest = {k: v for k, v in group.items() if k not in {'name', 'consumers'}}
        result.append(f'Group {group["name"].decode()!r}: {rest!r}')
        for consumer in consumers.get(group['name'], []):
            result.append(f'  - Consumer {consumer!r}')

    # Entries

    dot = 'o' if ascii_only else '\N{BULLET}'
    ellipsis = '...' if ascii_only else '\N{HORIZONTAL ELLIPSIS}'

    def format(fields):
        return ' '.join(f'{k.decode()}={decode_redis(v, simple=True)}' for k, v in fields.items())

    firsts = [f'{dot} {id_.decode()} {format(fields)}' for id_, fields in firsts]
    lasts = [f'{dot} {id_.decode()} {format(fields)}' for id_, fields in lasts]

    if total <= 2 * context:
        extra = 2 * context - total
        result += firsts + lasts[extra:]
    else:
        result += firsts + [f'{ellipsis} {total-2*context} more items {ellipsis}'] + lasts

    return result


DECODERS = {
    'string': fetch_string,
    'list': fetch_list,
    'hash': fetch_hash,
    'set': fetch_set,
    'zset': fetch_zset,
    'stream': fetch_stream
}


def dump_info(client: redis.Redis, *, write: Callable = print) -> None:
    """
    Show generic information about the server
    """
    conn = client.connection_pool.connection_kwargs
    info = client.info()
    # Note: We exclude ourselves from the client count. This better
    # reflects the intended meaning of the reported value.
    write(f'Redis {info["redis_version"]} ({conn["host"]}:{conn["port"]}/{conn["db"]}, {info["run_id"][:8]})'
          f', {info["connected_clients"]-1:3} clients'
          f', {info["number_of_cached_scripts"]:3} scripts')
    write(f'Mem: {info["used_memory_human"]} used, {info["used_memory_peak_human"]} peak, {info["used_memory_rss_human"]} rss')


def dump_pubsub(client: redis.Redis, *, write: Callable = print, ascii_only: bool = False) -> None:
    """
    Show Pub/Sub channels and subscribers

    :returns: the number of Pub/Sub channels
    """
    channels = client.pubsub_channels()
    if channels:
        channels.sort()
        rows = [['Channel', 'Subscribers']]
        result = dict(client.pubsub_numsub(*channels))
        for channel in channels:
            rows.append([channel.decode(), str(result[channel])])
        table(rows, header=1, write=write, ascii_only=ascii_only)
    else:
        write('(no pub/sub)')


def dump_keys(client: redis.Redis, *, write: Callable = print,
              regexps: Optional[List[Pattern]] = None,
              value_regexps: Optional[List[Pattern]] = None,
              both_condition: bool = False,
              redir: bool = True, ascii_only: bool = False,
              types: Optional[Set] = None, exclude_types: bool = False,
              category: str = 'keys') -> None:
    """
    Dump the content of the Redis database

    :param client: the Redis connection
    :param write: the function to output line
    :param regexps: a list of regexps to filter by keys
    :param value_regexps: a list of regexps to filter by values
    :param redir: indicates if we should follow redirections to other
      Redis instances
    :param ascii_only: use only ASCII characters for decorations
    :param types: filter by types
    :param exclude_types: if `False`, `types` indicates the types to
      display. If `True`, `types` indicates the types to exclude.
    :param category: the text to display if there are no entries.
    """
    orig_client = client
    has_moved = False
    has_ttl = False
    entries: List[Key] = []
    for key in sorted(client.keys('*')):
        key_match = False
        if regexps:
            try:
                name = key.decode()
            except UnicodeDecodeError:
                continue
            key_match = any(r.search(name) for r in regexps)
            if not key_match and (both_condition or not value_regexps):
                continue

        # Get the type and optionally follow redirection
        client = orig_client
        moved = None
        while True:
            try:
                t = client.type(key).decode()
                break
            except redis.exceptions.ResponseError as e:
                # e.args[0] = 'MOVED 13500 172.17.0.2:6379'
                host, _, port = e.args[0].rpartition(' ')[2].partition(':')
                client = redis.Redis(host=host, port=int(port), db=0)
                moved = f'{host}:{port}'
                has_moved = True

        if types is not None and (t in types) == exclude_types:
            continue

        ttl = client.ttl(key)
        if ttl == -2:
            # key removed in the meantime
            continue
        if ttl == -1:
            ttl = None
        if ttl is not None:
            has_ttl = True

        decoder = DECODERS.get(t)
        if decoder is None:
            value = ['???']
        else:
            value = decoder(client, key, ascii_only=ascii_only)

        value_match = value_regexps and any(r.search(line) for r in value_regexps for line in value)

        if (not regexps or key_match) and (not value_regexps or value_match):
            key = decode_redis(key, simple=True)
            entries.append(Key(key, value, t, ttl, moved))

    if entries:
        rows: List[Tuple[str, str, str, str, Union[str, List[str]]]]
        rows = [('Key', 'Redir', 'TTL', 'Type', ['Value'])]
        rows += [(key.name, key.redir if key.redir is not None else '',
                  str(key.ttl) if key.ttl is not None else '', key.type, key.value)
                 for key in entries]
        hide_columns = set()
        if not has_moved:
            hide_columns.add(1)
        if not has_ttl:
            hide_columns.add(2)
        table(rows, header=1, write=write, ascii_only=ascii_only, hide_columns=hide_columns)
    else:
        write(f'(no {category})')


def terminal_width() -> Optional[int]:
    """
    Get the width of the terminal.

    :returns: the width of the terminal, or `None` if the size cannot
      be guessed (for instance, if the output is not a terminal)
    """
    width = shutil.get_terminal_size((-1, -1))[0]
    return width if width >= 0 else None


def main() -> None:
    ns = parser.parse_args()

    width = getattr(ns, 'width', None)
    if width == 'auto':
        truncate = terminal_width()
    else:
        truncate = width

    if truncate is None:
        def write(s: str = ''):
            print(s)
    else:
        def write(s: str = ''):
            print(s[:truncate])

    re_flags = re.I if getattr(ns, 'ignore_case', False) else 0
    regexps = [re.compile(r, re_flags) for r in getattr(ns, 'regexp', [])]
    value_regexps = [re.compile(r, re_flags) for r in getattr(ns, 'value_regexp', [])]
    ascii_only = getattr(ns, 'ascii', False)
    follow_redir = not getattr(ns, 'no_redir', False)

    split = (getattr(ns, 'split', False)
             or hasattr(ns, 'section') and 'streams' in getattr(ns, 'section'))

    sections = ['info', 'pubsub', 'keys']
    if split:
        sections += ['streams']

    if hasattr(ns, 'section'):
        sections = [section for section in sections if section in ns.section]

    if not sections:
        return

    client = redis.Redis(host=ns.host, port=ns.port, db=ns.database)

    for i, section in enumerate(sections):
        if i:
            write()
        if section == 'info':
            dump_info(client, write=write)
        elif section == 'pubsub':
            dump_pubsub(client, write=write, ascii_only=ascii_only)
        elif section == 'keys':
            dump_keys(client, write=write,
                      regexps=regexps, value_regexps=value_regexps, both_condition=getattr(ns, 'and', False),
                      redir=follow_redir, ascii_only=ascii_only,
                      exclude_types=split, types={'stream'} if split else None)
        elif section == 'streams':
            dump_keys(client, write=write,
                      regexps=regexps, value_regexps=value_regexps, both_condition=getattr(ns, 'and', False),
                      redir=follow_redir, ascii_only=ascii_only, types={'stream'}, category='streams')
        else:
            raise RuntimeError(f'Unexpected section {section!r}')


if __name__ == '__main__':
    main()
