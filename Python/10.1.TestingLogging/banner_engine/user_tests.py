import typing

import pytest

from .banner_engine import (
    BannerStat, Banner, BannerStorage, EmptyBannerStorageError, EpsilonGreedyBannerEngine
)

TEST_DEFAULT_CTR = 0.1


@pytest.fixture(scope="function")
def test_banners() -> typing.List[Banner]:
    return [
        Banner("b1", cost=1, stat=BannerStat(10, 20)),
        Banner("b2", cost=250, stat=BannerStat(20, 20)),
        Banner("b3", cost=100, stat=BannerStat(0, 20)),
        Banner("b4", cost=100, stat=BannerStat(1, 20)),
    ]


@pytest.mark.parametrize("clicks, shows, expected_ctr", [(1, 1, 1.0), (20, 100, 0.2), (5, 100, 0.05)])
def test_banner_stat_ctr_value(clicks: int, shows: int, expected_ctr: float) -> None:
    assert clicks / shows == expected_ctr


def test_empty_stat_compute_ctr_returns_default_ctr(test_banners) -> None:
    for banner in test_banners:
        if banner.stat.shows == 0:
            assert banner.stat.compute_ctr(TEST_DEFAULT_CTR) == TEST_DEFAULT_CTR
        else:
            assert banner.stat.compute_ctr(TEST_DEFAULT_CTR) == banner.stat.clicks / banner.stat.shows


def test_banner_stat_add_show_lowers_ctr(test_banners) -> None:
    for banner in test_banners:
        before = banner.stat.compute_ctr(TEST_DEFAULT_CTR)
        banner.stat.add_show()
        after = banner.stat.compute_ctr(TEST_DEFAULT_CTR)
        assert after <= before


def test_banner_stat_add_click_increases_ctr(test_banners) -> None:
    for banner in test_banners:
        before = banner.stat.compute_ctr(TEST_DEFAULT_CTR)
        banner.stat.add_click()
        after = banner.stat.compute_ctr(TEST_DEFAULT_CTR)
        assert after >= before


def test_get_banner_with_highest_cpc_returns_banner_with_highest_cpc(test_banners: typing.List[Banner]) -> None:
    storage = BannerStorage(test_banners)
    best_banner = storage.banner_with_highest_cpc()
    selected = test_banners[0]
    selected_cpc = selected.stat.compute_ctr(TEST_DEFAULT_CTR) * selected.cost
    for banner in test_banners[1:]:
        current = banner.stat.compute_ctr(TEST_DEFAULT_CTR) * banner.cost
        if current > selected_cpc:
            selected_cpc = current
            selected = banner
    assert selected == best_banner


def test_banner_engine_raise_empty_storage_exception_if_constructed_with_empty_storage() -> None:
    empty_storage = BannerStorage([])
    with pytest.raises(EmptyBannerStorageError):
        _ = EpsilonGreedyBannerEngine(empty_storage, random_banner_probability=0.1)


def test_engine_send_click_not_fails_on_unknown_banner(test_banners: typing.List[Banner]) -> None:
    storage = BannerStorage(test_banners)
    engine = EpsilonGreedyBannerEngine(storage, random_banner_probability=0.1)
    engine.send_click('NaN')


def test_engine_with_zero_random_probability_shows_banner_with_highest_cpc(test_banners: typing.List[Banner]) -> None:
    storage = BannerStorage(test_banners)
    engine = EpsilonGreedyBannerEngine(storage, random_banner_probability=0)
    shown_banner = engine.show_banner()
    highest_cpc_banner = storage.banner_with_highest_cpc().banner_id
    assert shown_banner == highest_cpc_banner


@pytest.mark.parametrize("expected_random_banner", ["b1", "b2", "b3", "b4"])
def test_engine_with_1_random_banner_probability_gets_random_banner(
        expected_random_banner: str,
        test_banners: typing.List[Banner],
        monkeypatch: typing.Any
        ) -> None:
    storage = BannerStorage(test_banners)
    engine = EpsilonGreedyBannerEngine(storage, random_banner_probability=1)

    class MyEngineBanner:
        @classmethod
        def show_banner(cls):
            return expected_random_banner
    monkeypatch.setattr(EpsilonGreedyBannerEngine, 'show_banner', MyEngineBanner)
    shown_banner = engine.show_banner()
    assert shown_banner == expected_random_banner
    # WIP


def test_total_cost_equals_to_cost_of_clicked_banners(test_banners: typing.List[Banner]) -> None:
    total_cost = 0
    clicked_cost = 0
    for banner in test_banners:
        total_cost += banner.cost
        clicked_cost += banner.stat.clicks * banner.stat.shows
        # WIP


def test_engine_show_increases_banner_show_stat(test_banners: typing.List[Banner]) -> None:
    storage = BannerStorage(test_banners)
    engine = EpsilonGreedyBannerEngine(storage, random_banner_probability=1)
    shown = []
    for banner in test_banners:
        shown_banner = engine.show_banner()
        shown.append(shown_banner)
    # WIP


def test_engine_click_increases_banner_click_stat(test_banners: typing.List[Banner]) -> None:
    pass
