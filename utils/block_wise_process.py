import logging

from pymongo import MongoClient
from subprocess import check_call, CalledProcessError


def check_block(block, db_host, db_name, collection_name):
    '''
    Check if a block has been completed by querying MongoDB.

    Used by daisy as a check_function to determine if a block has already been processed.
    Queries MongoDB collection for the block's ID to verify completion status.

    Parameters
    ----------
    block : daisy.Block
        Block object containing block_id to check.
    db_host : str
        MongoDB connection URI.
    db_name : str
        MongoDB database name.
    collection_name : str
        MongoDB collection name tracking block completion (e.g., 'blocks_predicted',
        'blocks_fragments', 'blocks_agglomerated_*').

    Returns
    -------
    bool
        True if block is marked as completed in MongoDB, False otherwise.

    Notes
    -----
    - Block is considered done if at least one document with matching block_id exists
    - Workers write completion documents to collection after processing
    '''

    client = MongoClient(db_host)
    db = client[db_name]
    blocks_collection = db[collection_name]
    done = blocks_collection.count_documents({'block_id': block.block_id}) >= 1

    return done


def daisy_call(command, log_out, log_err):
    '''
    Run ``command`` in a subprocess, log stdout and stderr to ``log_out``
    and ``log_err``
    Copied from older version of daisy.
    '''

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
