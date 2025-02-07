import logging

from pymongo import MongoClient
from subprocess import check_call, CalledProcessError


def check_block(block, db_host, db_name, collection_name):

    client = MongoClient(db_host)
    db = client[db_name]
    blocks_collection = db[collection_name]
    done = blocks_collection.count_documents({'block_id': block.block_id}) >= 1

    return done


def daisy_call(command, log_out, log_err):
    """
    Run ``command`` in a subprocess, log stdout and stderr to ``log_out``
    and ``log_err``
    Copied from older version of daisy.
    """

    logger = logging.getLogger(__name__)
    logger.debug(
        "Running subprocess with:"
        "\n\tcommand %s"
        "\n\tlog_out %s"
        "\n\tlog_err %s",
        command, log_out, log_err)
    try:
        with open(log_out, 'w') as stdout:
            with open(log_err, 'w') as stderr:
                check_call(
                    ' '.join(command),
                    shell=True,
                    stdout=stdout,
                    stderr=stderr)

    except CalledProcessError as exc:
        raise Exception(
            "Calling %s failed with return code %s, stderr in %s" %
            (' '.join(command), exc.returncode, stderr.name))
    except KeyboardInterrupt:
        raise Exception("Canceled by SIGINT")
