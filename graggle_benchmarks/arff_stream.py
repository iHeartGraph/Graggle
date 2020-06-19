import arff
from arff import _get_data_object_for_decoding, _TK_DESCRIPTION, _TK_RELATION, _TK_DATA, _TK_ATTRIBUTE, _TK_COMMENT
from arff import *

class ArffStreamer(arff.ArffDecoder):
    def __init__(self):
        super().__init__()
        
    def decode(self, s, encode_nominal=False, matrix_type=DENSE):
        '''Do the job the ``encode``.'''

        # Make sure this method is idempotent
        self._current_line = 0

        # If string, convert to a list of lines
        if isinstance(s, basestring):
            s = s.strip('\r\n ').replace('\r\n', '\n').split('\n')

        # Create the return object
        obj = {
            u'description': u'',
            u'relation': u'',
            u'attributes': [],
            u'data': []
        }
        attribute_names = {}

        # Create the data helper object
        data = _get_data_object_for_decoding(matrix_type)

        # Read all lines
        STATE = _TK_DESCRIPTION
        s = iter(s)
        for row in s:
            self._current_line += 1
            # Ignore empty lines
            row = row.strip(' \r\n')
            if not row: continue

            u_row = row.upper()

            # DESCRIPTION -----------------------------------------------------
            if u_row.startswith(_TK_DESCRIPTION) and STATE == _TK_DESCRIPTION:
                obj['description'] += self._decode_comment(row) + '\n'
            # -----------------------------------------------------------------

            # RELATION --------------------------------------------------------
            elif u_row.startswith(_TK_RELATION):
                if STATE != _TK_DESCRIPTION:
                    raise BadLayout()

                STATE = _TK_RELATION
                obj['relation'] = self._decode_relation(row)
            # -----------------------------------------------------------------

            # ATTRIBUTE -------------------------------------------------------
            elif u_row.startswith(_TK_ATTRIBUTE):
                if STATE != _TK_RELATION and STATE != _TK_ATTRIBUTE:
                    raise BadLayout()

                STATE = _TK_ATTRIBUTE

                attr = self._decode_attribute(row)
                if attr[0] in attribute_names:
                    raise BadAttributeName(attr[0], attribute_names[attr[0]])
                else:
                    attribute_names[attr[0]] = self._current_line
                obj['attributes'].append(attr)

                if isinstance(attr[1], (list, tuple)):
                    if encode_nominal:
                        conversor = EncodedNominalConversor(attr[1])
                    else:
                        conversor = NominalConversor(attr[1])
                else:
                    CONVERSOR_MAP = {'STRING': unicode,
                                     'INTEGER': lambda x: int(float(x)),
                                     'NUMERIC': float,
                                     'REAL': float}
                    conversor = CONVERSOR_MAP[attr[1]]

                self._conversors.append(conversor)
            # -----------------------------------------------------------------

            # DATA ------------------------------------------------------------
            elif u_row.startswith(_TK_DATA):
                if STATE != _TK_ATTRIBUTE:
                    raise BadLayout()

                break
            # -----------------------------------------------------------------

            # COMMENT ---------------------------------------------------------
            elif u_row.startswith(_TK_COMMENT):
                pass
            # -----------------------------------------------------------------
            
            else:
                # Never found @DATA
                raise BadLayout()


        for row in s:
            self._current_line += 1
            row = row.strip()
            # Ignore empty lines and comment lines.
            if row and not row.startswith(_TK_COMMENT):
                d = {}
                row = row.split(',')
                
                for i in range(len(row)-1):
                    if int(row[i]) > 0:
                        d[i] = int(row[i])
                
                d['class'] = row[-1]
                yield d
        
        
                
if __name__ == '__main__':
    import json 
    
    DATA = '/mnt/raid0_24TB/datasets/ContextGraph/'
    REUTERS = DATA + 'Reuters-21578.arff'
    
    de = ArffStreamer()
    for d in de.decode(open(REUTERS, 'r')):
        print(json.dumps(d, indent=4))