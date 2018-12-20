"""Database backend for experimental data."""
import typing

import sqlalchemy as sql
from absl import app
from absl import flags
from sqlalchemy.ext import declarative

from experimental.compilers.reachability import reachability_pb2
from labm8 import sqlutil


FLAGS = flags.FLAGS

Base = declarative.declarative_base()


class LlvmBytecode(Base, sqlutil.ProtoBackedMixin,
                   sqlutil.TablenameFromClassNameMixin):
  """A table of Llvm bytecodes."""
  proto_t = reachability_pb2.LlvmBytecode

  id: int = sql.Column(sql.Integer, primary_key=True)

  source_name: str = sql.Column(sql.String(256), nullable=False)
  relpath: str = sql.Column(sql.String(256), nullable=False)
  lang: str = sql.Column(sql.String(32), nullable=False)
  cflags: str = sql.Column(sql.String(1024), nullable=False)
  charcount: int = sql.Column(sql.Integer, nullable=False)
  linecount: int = sql.Column(sql.Integer, nullable=False)
  bytecode: str = sql.Column(
      sql.UnicodeText().with_variant(sql.UnicodeText(2 ** 31), 'mysql'),
      nullable=False)
  clang_returncode: int = sql.Column(sql.Integer, nullable=False)
  error_message: str = sql.Column(
      sql.UnicodeText().with_variant(sql.UnicodeText(2 ** 31), 'mysql'))

  def SetProto(self, proto: proto_t) -> None:
    raise NotImplementedError

  @classmethod
  def FromProto(cls, proto: proto_t) -> typing.Dict[str, typing.Any]:
    """Return a dictionary of instance constructor args from proto."""
    return {
      'source_name': proto.source_name,
      'relpath': proto.relpath,
      'lang': proto.lang,
      'cflags': proto.cflags,
      'charcount': len(proto.bytecode),
      'linecount': len(proto.bytecode.split('\n')),
      'bytecode': proto.bytecode,
      'clang_returncode': proto.clang_returncode,
      'error_message': proto.error_message,
    }


class Database(sqlutil.Database):

  def __init__(self, url: str):
    super(Database, self).__init__(url, Base)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))


if __name__ == '__main__':
  app.run(main)
