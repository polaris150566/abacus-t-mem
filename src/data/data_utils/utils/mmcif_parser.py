import collections
import dataclasses
import functools
import io
from typing import Any, Mapping, Optional, Sequence, Tuple

import logging
from Bio import PDB
try:
  from Bio.Data import SCOPData
except:
  from Bio.Data import PDBData

logger = logging.getLogger(__name__)

# Type aliases:
ChainId = str
PdbHeader = Mapping[str, Any]
PdbStructure = PDB.Structure.Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]

def mmcif_loop_to_list(prefix: str,
                       parsed_info: MmCIFDict) -> Sequence[Mapping[str, str]]:
  """Extracts loop associated with a prefix from mmCIF data as a list.

  Reference for loop_ in mmCIF:
    http://mmcif.wwpdb.org/docs/tutorials/mechanics/pdbx-mmcif-syntax.html

  Args:
    prefix: Prefix shared by each of the data items in the loop.
      e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
      _entity_poly_seq.mon_id. Should include the trailing period.
    parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
      parser.

  Returns:
    Returns a list of dicts; each dict represents 1 entry from an mmCIF loop.
  """
  cols = []
  data = []
  for key, value in parsed_info.items():
    if key.startswith(prefix):
      cols.append(key)
      data.append(value)

  assert all([len(xs) == len(data[0]) for xs in data]), (
      'mmCIF error: Not all loops are the same length: %s' % cols)

  return [dict(zip(cols, xs)) for xs in zip(*data)]


def mmcif_loop_to_dict(prefix: str,
                       index: str,
                       parsed_info: MmCIFDict,
                       ) -> Mapping[str, Mapping[str, str]]:
  """Extracts loop associated with a prefix from mmCIF data as a dictionary.

  Args:
    prefix: Prefix shared by each of the data items in the loop.
      e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
      _entity_poly_seq.mon_id. Should include the trailing period.
    index: Which item of loop data should serve as the key.
    parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
      parser.

  Returns:
    Returns a dict of dicts; each dict represents 1 entry from an mmCIF loop,
    indexed by the index column.
  """
  entries = mmcif_loop_to_list(prefix, parsed_info)
  return {entry[index]: entry for entry in entries}



def get_release_date(parsed_info: MmCIFDict) -> str:
  """Returns the oldest revision date."""
  revision_dates = parsed_info['_pdbx_audit_revision_history.revision_date']
  return min(revision_dates)


def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
  """Returns a basic header containing method, release date and resolution."""
  header = {}
  header['release_date'] = None
  header['resolution'] = 0.00
  header['structure_method'] = None

  experiments = mmcif_loop_to_list('_exptl.', parsed_info)
  header['structure_method'] = ','.join([
      experiment['_exptl.method'].lower() for experiment in experiments])

  # Note: The release_date here corresponds to the oldest revision. We prefer to
  # use this for dataset filtering over the deposition_date.
  if '_pdbx_audit_revision_history.revision_date' in parsed_info:
    header['release_date'] = get_release_date(parsed_info)
  else:
    logger.debug('Could not determine release_date: %s',
                    parsed_info['_entry.id'])

  for res_key in ('_refine.ls_d_res_high', '_em_3d_reconstruction.resolution',
                  '_reflns.d_resolution_high'):
    if res_key in parsed_info:
      try:
        raw_resolution = parsed_info[res_key][0]
        header['resolution'] = float(raw_resolution)
      except ValueError:
        logger.debug('Invalid resolution format: %s', parsed_info[res_key])

  return header
