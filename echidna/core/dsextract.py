import math
import rat
import echidna.calc.constants as const


def function_factory(dimension, **kwargs):
    '''Factory function that returns a dsextract class
    corresponding to the dimension (i.e. DS parameter)
    that is being extracted.

    Args:
      dimension (str): to extract from a RAT DS/ntuple file.
      \**kwargs (dict): to be passed to and checked by the extractor.

    Raises:
      IndexError: dimension is an unknown parameter

    Retuns:
      Extractor object.
    '''
    if dimension == "energy_mc":
        return EnergyExtractMC()
    elif dimension == "energy_reco":
        return EnergyExtractReco()
    elif dimension == "energy_truth":
        return EnergyExtractTruth()
    elif dimension == "radial_mc":
        return RadialExtractMC()
    elif dimension == "radial_reco":
        return RadialExtractReco()
    elif dimension == "radial3_mc":
        return Radial3ExtractMC(**kwargs)
    elif dimension == "radial3_reco":
        return Radial3ExtractReco(**kwargs)
    else:
        raise IndexError("Unknown parameter: %s" % dimension)


class Extractor(object):
    '''Base class for extractor classes.

    Args:
      name (str): of the dimension

    Attributes:
      _name (str): of the dimension
    '''

    def __init__(self, name):
        '''Initialise the class
        '''
        self.name = name


class EnergyExtractMC(Extractor):
    '''Quenched energy extraction methods.
    '''

    def __init__(self):
        '''Initialise the class
        '''
        super(EnergyExtractMC, self).__init__("energy_mc")

    def get_valid_root(self, mc):
        '''Check whether energy of a DS::MC is valid

        Args:
          mc (:class:`RAT.DS.MC`) entry

        Returns:
          bool: Validity boolean
        '''
        if mc.GetMCParticleCount > 0:
            return True
        return False

    def get_value_root(self, mc):
        '''Get energy value from a DS::MC

        Args:
          mc (:class:`RAT.DS.MC`) entry

        Returns:
          float: True quenched energy
        '''
        return mc.GetScintQuenchedEnergyDeposit()

    def get_valid_ntuple(self, entry):
        '''Check whether energy of an ntuple MC is valid

        Args:
          entry (:class:`ROOT.TChain`) chain entry

        Returns:
          bool: Validity boolean
        '''
        if entry.mc == 1:
            return True
        return False

    def get_value_ntuple(self, entry):
        '''Get energy value from an ntuple MC

        Args:
          entry (:class:`ROOT.TChain`) chain entry

        Returns:
          float: True quenched energy
        '''
        return entry.mcEdepQuenched


class EnergyExtractReco(Extractor):
    '''Reconstructed energy extraction methods.
    '''

    def __init__(self):
        '''Initialise the class
        '''
        super(EnergyExtractReco, self).__init__("energy_reco")

    def get_valid_root(self, ev):
        '''Check whether energy of a DS::EV is valid

        Args:
          ev (:class:`RAT.DS.EV`) event

        Returns:
          bool: Validity boolean
        '''
        if ev.DefaultFitVertexExists() and \
                ev.GetDefaultFitVertex().ContainsEnergy() \
                and ev.GetDefaultFitVertex().ValidEnergy():
            return True
        return False

    def get_value_root(self, ev):
        '''Get energy value from a DS::EV

        Args:
          ev (:class:`RAT.DS.EV`) event

        Returns:
          float: Reconstructed energy
        '''
        return ev.GetDefaultFitVertex().GetEnergy()

    def get_valid_ntuple(self, entry):
        '''Check whether energy of an ntuple EV is valid

        Args:
          entry (:class:`ROOT.TChain`) chain entry

        Returns:
          bool: Validity boolean
        '''
        return (entry.scintFit != 0 and entry.energy > 0)

    def get_value_ntuple(self, entry):
        '''Get energy value from an ntuple EV

        Args:
          entry (:class:`ROOT.TChain`) chain entry

        Returns:
          float: Reconstructed energy
        '''
        return entry.energy


class EnergyExtractTruth(Extractor):
    '''True MC energy extraction methods.
    '''

    def __init__(self):
        '''Initialise the class
        '''
        super(EnergyExtractTruth, self).__init__("energy_truth")

    def get_valid_root(self, mc):
        '''Check whether energy of a DS::MC is valid

        Args:
          mc (:class:`RAT.DS.MC`) entry

        Returns:
          bool: Validity boolean
        '''
        if mc.GetMCParticleCount > 0:
            return True
        return False

    def get_value_root(self, mc):
        '''Get energy value from a DS::MC

        Args:
          mc (:class:`RAT.DS.MC`) entry

        Returns:
          float: True energy
        '''
        return mc.GetScintEnergyDeposit()

    def get_valid_ntuple(self, entry):
        '''Check whether energy of an ntuple MC is valid

        Args:
          entry (:class:`ROOT.TChain`) chain entry

        Returns:
          bool: Validity boolean
        '''
        if entry.mc == 1:
            return True
        return False

    def get_value_ntuple(self, entry):
        '''Get energy value from an ntuple MC

        Args:
          entry (:class:`ROOT.TChain`) chain entry

        Returns:
          float: True energy
        '''
        entry.mcEdep


class RadialExtractMC(Extractor):
    '''True radial extraction methods.
    '''

    def __init__(self):
        '''Initialise the class
        '''
        super(RadialExtractMC, self).__init__("radial_mc")

    def get_valid_root(self, mc):
        '''Check whether radius of a DS::MC is valid

        Args:
          ev (:class:`RAT.DS.MC`) event

        Returns:
          bool: Validity boolean
        '''
        if mc.GetMCParticleCount > 0:
            return True
        return False

    def get_value_root(self, mc):
        '''Get radius value from a DS::MC

        Args:
          ev (:class:`RAT.DS.MC`) event

        Returns:
          float: True radius
        '''
        return mc.GetMCParticle(0).GetPosition().Mag()

    def get_valid_ntuple(self, entry):
        '''Check whether energy of an ntuple MC is valid

        Args:
          ev (:class:`ROOT.TChain`) chain entry

        Returns:
          bool: Validity boolean
        '''
        if entry.mc == 1:
            return True
        return False

    def get_value_ntuple(self, entry):
        '''Get radius value from an ntuple MC

        Args:
          ev (:class:`ROOT.TChain`) chain entry

        Returns:
          float: True radius
        '''
        return math.fabs(math.sqrt((entry.mcPosx)**2 +
                                   (entry.mcPosy)**2 +
                                   (entry.mcPosz)**2))


class RadialExtractReco(Extractor):
    '''Reconstructed radial extraction methods.
    '''

    def __init__(self):
        '''Initialise the class
        '''
        super(RadialExtractReco, self).__init__("radial_reco")

    def get_valid_root(self, ev):
        '''Check whether radius of a DS::EV is valid

        Args:
          ev (:class:`RAT.DS.EV`) event

        Returns:
          bool: Validity boolean
        '''
        if ev.DefaultFitVertexExists() and \
                ev.GetDefaultFitVertex().ContainsPosition() \
                and ev.GetDefaultFitVertex().ValidPosition():
            return True
        return False

    def get_value_root(self, ev):
        '''Get radius value from a DS::EV

        Args:
          ev (:class:`RAT.DS.EV`) event

        Returns:
          float: Reconstructed radius
        '''
        return ev.GetDefaultFitVertex().GetPosition().Mag()

    def get_valid_ntuple(self, entry):
        '''Check whether radius of an ntuple EV is valid

        Args:
          ev (:class:`ROOT.TChain`) chain entry

        Returns:
          bool: Validity boolean
        '''
        return entry.scintFit != 0

    def get_value_ntuple(self, entry):
        '''Get radius value from an ntuple EV

        Args:
          ev (:class:`ROOT.TChain`) chain entry

        Returns:
          float: Reconstructed radius
        '''
        return math.fabs(math.sqrt((entry.posx)**2 +
                                   (entry.posy)**2 +
                                   (entry.posz)**2))


class Radial3ExtractMC(Extractor):
    ''' True :math:`(radius/av_radius)^3`radial extraction methods.
    '''

    def __init__(self, av_radius=None):
        '''Initialise the class
        '''
        super(Radial3ExtractMC, self).__init__("radial3_mc")
        if av_radius:
            self._av_radius = av_radius
        else:
            self._av_radius = const._av_radius

    def get_valid_root(self, mc):
        '''Check whether radius of a DS::MC is valid

        Args:
          ev (:class:`RAT.DS.MC`) event

        Returns:
          bool: Validity boolean
        '''
        if mc.GetMCParticleCount > 0:
            return True
        return False

    def get_value_root(self, mc):
        '''Get radius value from a DS::MC

        Args:
          ev (:class:`RAT.DS.MC`) event

        Returns:
          float: True :math:`(radius/av_radius)^3`
        '''
        return (mc.GetMCParticle(0).GetPosition().Mag() / self._av_radius) ** 3

    def get_valid_ntuple(self, entry):
        '''Check whether energy of an ntuple MC is valid

        Args:
          ev (:class:`ROOT.TChain`) chain entry

        Returns:
          bool: Validity boolean
        '''
        if entry.mc == 1:
            return True
        return False

    def get_value_ntuple(self, entry):
        '''Get radius value from an ntuple MC

        Args:
          ev (:class:`ROOT.TChain`) chain entry

        Returns:
          float: True :math:`(radius/av_radius)^3`
        '''
        return (math.fabs(math.sqrt((entry.mcPosx)**2 +
                                    (entry.mcPosy)**2 +
                                    (entry.mcPosz)**2)) / self._av_radius) ** 3


class Radial3ExtractReco(Extractor):
    ''' Reconstructed :math:`(radius/av_radius)^3` radial extraction methods.
    '''

    def __init__(self, av_radius=None):
        '''Initialise the class
        '''
        super(Radial3ExtractReco, self).__init__("radial3_reco")
        if av_radius:
            self._av_radius = av_radius
        else:
            self._av_radius = const._av_radius

    def get_valid_root(self, ev):
        '''Check whether radius of a DS::EV is valid

        Args:
          ev (:class:`RAT.DS.EV`) event

        Returns:
          bool: Validity boolean
        '''
        if ev.DefaultFitVertexExists() and \
                ev.GetDefaultFitVertex().ContainsPosition() \
                and ev.GetDefaultFitVertex().ValidPosition():
            return True
        return False

    def get_value_root(self, ev):
        '''Get radius value from a DS::EV

        Args:
          ev (:class:`RAT.DS.EV`) event

        Returns:
          float: Reconstructed :math:`(radius/av_radius)^3`
        '''
        return (ev.GetDefaultFitVertex().GetPosition().Mag() /
                self._av_radius) ** 3

    def get_valid_ntuple(self, entry):
        '''Check whether radius of an ntuple EV is valid

        Args:
          ev (:class:`ROOT.TChain`) chain entry

        Returns:
          bool: Validity boolean
        '''
        return entry.scintFit != 0

    def get_value_ntuple(self, entry):
        '''Get radius value from an ntuple EV

        Args:
          ev (:class:`ROOT.TChain`) chain entry

        Returns:
          float: Reconstructed :math:`(radius/av_radius)^3`
        '''
        return (math.fabs(math.sqrt((entry.posx)**2 +
                                    (entry.posy)**2 +
                                    (entry.posz)**2)) / self._av.radius) ** 3
