"""Run kernels in features database using CGO'17 driver and settings."""
import collections
import typing

from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db as db
from gpu.cldrive import api as cldrive
from gpu.cldrive.legacy import env as cldrive_env
from gpu.cldrive.proto import cldrive_pb2
from labm8 import app
from labm8 import pbutil
from labm8 import sqlutil
from labm8 import system
from research.cummins_2017_cgo import opencl_kernel_driver

FLAGS = app.FLAGS

app.DEFINE_string(
    'db',
    'sqlite:///tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db',
    'URL of the database to load static features from, and store dynamic '
    'features to.')
app.DEFINE_string(
    'env', 'Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2',
    'The OpenCL environment to execute benchmark suites on. To list the '
    'available environments, run `bazel run //gpu/clinfo`.')
app.DEFINE_integer('num_runs', 30, 'The number of runs for each benchmark.')
app.DEFINE_integer('batch_size', 16,
                   'The number of kernels to process at a time.')

KernelToDrive = collections.namedtuple('KernelToDrive', ['id', 'src'])

# Use the same combinations of local and global sizes as in the CGO'17 paper.
LSIZE_GSIZE_PROTO_PAIRS = [
    cldrive_pb2.DynamicParams(global_size_x=y, local_size_x=x)
    for x, y in opencl_kernel_driver.LSIZE_GSIZE_PAIRS
]


def GetBatchOfKernelsToDrive(session: sqlutil.Session,
                             env: cldrive_env.OpenCLEnvironment,
                             batch_size: int):
  """Get a batch of kernels to run."""
  already_done = session.query(db.DynamicFeatures.static_features_id) \
    .filter(db.DynamicFeatures.opencl_env == env.name)
  q = session.query(
      db.StaticFeatures.id, db.StaticFeatures.src) \
    .filter(~db.StaticFeatures.id.in_(already_done)) \
    .limit(batch_size)
  return [KernelToDrive(*row) for row in q]


def DriveBatchAndRecordResults(session: sqlutil.Session,
                               batch: typing.List[KernelToDrive],
                               env: cldrive_env.OpenCLEnvironment) -> None:
  """Drive a batch of kernels and record dynamic features."""
  try:
    instances = cldrive.DriveToDataFrame(
        cldrive_pb2.CldriveInstances(instance=[
            cldrive_pb2.CldriveInstance(
                opencl_env=env.proto,
                opencl_src=src,
                min_runs_per_kernel=FLAGS.num_runs,
                dynamic_params=LSIZE_GSIZE_PROTO_PAIRS,
            ) for _, src in batch
        ]))
    print(instances)
    import sys
    sys.exit(0)
    if len(instances.instance) != len(batch):
      raise OSError(f"Number of instances ({len(instances.instance)}) != "
                    f"batch size ({len(batch)})")

    for (static_features_id, _), instance in zip(batch, instances.instance):
      if len(instance.kernel) < 1:
        session.add(
            db.DynamicFeatures(
                static_features_id=static_features_id,
                opencl_env=env.name,
                hostname=system.HOSTNAME,
                result=cldrive_pb2.CldriveInstance.InstanceOutcome.Name(
                    instance.outcome),
            ))
      else:
        if len(instance.kernel) != 1:
          raise OSError(f"{instance.kernel} kernels found!")

        result = cldrive_pb2.CldriveInstance.InstanceOutcome.Name(
            instance.outcome)
        if result == 'PASS':
          result = cldrive_pb2.CldriveKernelInstance.KernelInstanceOutcome.Name(
              instance.kernel[0].outcome)

        session.add(
            db.DynamicFeatures(
                static_features_id=static_features_id,
                opencl_env=env.name,
                hostname=system.HOSTNAME,
                result=result,
            ))

        for run in instance.kernel[0].run:
          session.add_all([
              db.DynamicFeatures(
                  static_features_id=static_features_id,
                  opencl_env=env.name,
                  hostname=system.HOSTNAME,
                  dataset=f'{log.global_size},{log.local_size}',
                  gsize=log.global_size,
                  wgsize=log.local_size,
                  transferred_bytes=log.transferred_bytes,
                  runtime_ms=log.runtime_ms,
              ) for log in run.log
          ])
  except pbutil.ProtoWorkerTimeoutError:
    session.add_all([
        db.DynamicFeatures(
            static_features_id=static_features_id,
            opencl_env=env.name,
            hostname=system.HOSTNAME,
            result='DRIVER_TIMEOUT',
        ) for (static_features_id, _) in batch
    ])


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  database = db.Database(FLAGS.db)
  env = cldrive_env.OpenCLEnvironment.FromName(FLAGS.env)

  batch_num = 0
  while True:
    batch_num += 1
    app.Log(1, 'Batch %d', batch_num)
    with database.Session(commit=True) as session:
      batch = GetBatchOfKernelsToDrive(session, env, FLAGS.batch_size)
      if not batch:
        app.Log(1, 'Done. Nothing more to run!')
        return

      DriveBatchAndRecordResults(session, batch, env)


if __name__ == '__main__':
  app.RunWithArgs(main)
